# Suppress OpenCV TIFF warnings
import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import argparse
import logging
import os
import random
import json
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from segment_anything import sam_model_registry

from randall_pipeline.segmentation_model import DoRA_Sam
from randall_pipeline.segmentation_trainer import trainer_hm
from randall_pipeline.segmentation_dataset import hm_dataset_multiclass

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True,
                    help='Preprocessed root written by preprocess_segmentation_data.py')
parser.add_argument('--output', type=str, default='outputs/segmentation-training')
parser.add_argument('--dataset', type=str,
                    default='randall', help='experiment name')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output classes')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch size per GPU')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=1024, help='model input size')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_l', help='SAM backbone; paper recipe uses vit_l')
parser.add_argument('--ckpt', type=str, required=True,
                    help='Separately obtained Meta SAM ViT-L checkpoint')
parser.add_argument('--dora_ckpt', type=str, default=None, help='Finetuned dora checkpoint')
parser.add_argument('--rank', type=int, default=8, help='DoRA rank')
parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True, help='Warm up the learning rate')
parser.add_argument('--warmup_period', type=int, default=750,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action=argparse.BooleanOptionalAction, default=True, help='Use AdamW')
parser.add_argument('--dice_param', type=float, default=0.6)
parser.add_argument('--hint_loss_weight', type=float, default=3.0,
                    help='Multiplier applied to hint losses (focal_hint + dice_hint).')
parser.add_argument('--id2label', type=str, default=None,
                    help='Path to JSON file mapping class IDs to label names (for logging). Format: {"0": "background", "1": "class1", ...} or id2label.json format.')
parser.add_argument('--test_subset_size', type=int, default=None,
                    help='Number of test samples to evaluate (e.g., 50). If None, evaluates on entire test set.')
parser.add_argument('--background_weight', type=float, default=1.0,
                    help='Multiplier applied to background class (ID 0) weight in Dice loss.')
parser.add_argument('--test_eval_interval', type=int, default=5,
                    help='Evaluate test loss every N epochs (default: 5).')
parser.add_argument('--cache_dir', type=str, default=".cache_multiclass",
                    help='Directory for on-disk cached, resized tensors (relative to repo). Set empty to disable.')
parser.add_argument('--cache_read', action=argparse.BooleanOptionalAction, default=True,
                    help='Read cached tensors if present.')
parser.add_argument('--cache_write', action=argparse.BooleanOptionalAction, default=True,
                    help='Write cached tensors if missing.')
parser.add_argument('--num_workers', type=int, default=16,
                    help='DataLoader worker processes.')
parser.add_argument('--prefetch_factor', type=int, default=4,
                    help='DataLoader prefetch_factor when num_workers > 0.')
parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                    help='Keep DataLoader workers alive between epochs.')
parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                    help='Pin CPU memory for faster host to GPU transfer.')
parser.add_argument('--gpu_aug', action=argparse.BooleanOptionalAction, default=True,
                    help='Apply random rot/flip augmentations on GPU.')
parser.add_argument('--spgen_variant', type=str, default='depthwise',
                    choices=['original_head', 'depthwise'],
                    help='SPGen head variant to use.')
args = parser.parse_args()
if args.id2label is None:
    candidate = Path(args.root_path) / "id2label.json"
    if candidate.is_file():
        args.id2label = str(candidate)

def _load_id2label(id2label_path):
    if not id2label_path:
        return None
    path = Path(id2label_path)
    if not path.exists():
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    return {i: str(v) for i, v in enumerate(data)}


def _compute_class_frequencies_and_weights(root_path, num_classes, id2label=None, background_weight=1.0):
    dataset = hm_dataset_multiclass(base_dir=root_path, split="train", transform=None, id2label=id2label)
    counts = np.zeros(num_classes, dtype=np.int64)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample['label']
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if label.min() < 0 or label.max() >= num_classes:
            print(f"Warning: label values out of range in sample {idx}; clipping to [0, {num_classes - 1}]")
            label = np.clip(label, 0, num_classes - 1)
        counts += np.bincount(label.ravel(), minlength=num_classes)

    total = counts.sum()
    if total == 0:
        freqs = np.zeros(num_classes, dtype=np.float64)
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        freqs = counts / total
        weights = 1.0 / (freqs + 1e-6)
        if num_classes > 0:
            weights[0] *= background_weight
        weights = weights / weights.sum() * num_classes

    print("Class frequencies (train split):")
    for class_id in range(num_classes):
        class_name = id2label.get(class_id, f"Class {class_id}") if id2label else f"Class {class_id}"
        print(f"  {class_name} (ID {class_id}): freq={freqs[class_id]:.6f}, count={counts[class_id]}")
    print("Inverse-frequency weights (normalized):")
    for class_id in range(num_classes):
        class_name = id2label.get(class_id, f"Class {class_id}") if id2label else f"Class {class_id}"
        print(f"  {class_name} (ID {class_id}): weight={weights[class_id]:.6f}")

    return freqs, weights


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        dataset_name: {
            'root_path': args.root_path,
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1],
                                                                spgen_variant=args.spgen_variant)

    if not torch.cuda.is_available():
        raise RuntimeError("Segmentation training requires CUDA")
    net = DoRA_Sam(sam, args.rank).cuda()

    if args.dora_ckpt is not None:
        net.load_dora_parameters(args.dora_ckpt)

    # For multiclass, always use multimask_output if num_classes > 1
    if args.num_classes > 1:
        multimask_output = True
        print(f"Multiclass training with {args.num_classes} classes - using multimask_output=True")
    else:
        multimask_output = False
        print("Binary training - using multimask_output=False")
    
    # Validate num_classes
    if args.num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {args.num_classes}")

    low_res = img_embedding_size * 4
    args.low_res = low_res

    id2label = _load_id2label(args.id2label)
    _, class_weights = _compute_class_frequencies_and_weights(
        args.root_path,
        args.num_classes,
        id2label=id2label,
        background_weight=args.background_weight,
    )
    args.class_weights = class_weights.tolist()

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {dataset_name: trainer_hm}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
