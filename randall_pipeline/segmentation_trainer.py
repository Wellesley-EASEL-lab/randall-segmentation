import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import json
from pathlib import Path


def gpu_random_rot_flip(image, label, low_res_label):
    # Applies random rot90 + flip per-sample on GPU
    b = image.shape[0]
    for i in range(b):
        k = int(torch.randint(0, 4, (1,), device=image.device).item())
        if k:
            image[i] = torch.rot90(image[i], k, dims=(1, 2))
            label[i] = torch.rot90(label[i], k, dims=(0, 1))
            low_res_label[i] = torch.rot90(low_res_label[i], k, dims=(0, 1))
        if torch.rand(1, device=image.device) > 0.5:
            axis = int(torch.randint(1, 3, (1,), device=image.device).item())
            image[i] = torch.flip(image[i], dims=(axis,))
            label_axis = axis - 1
            label[i] = torch.flip(label[i], dims=(label_axis,))
            low_res_label[i] = torch.flip(low_res_label[i], dims=(label_axis,))
    return image, label, low_res_label


def dice_loss_multiclass(logits, target, num_classes, smooth=1e-6, class_weights=None):
    """
    Multiclass Dice loss.
    
    Args:
        logits: Tensor of shape (B, C, H, W) where C is num_classes
        target: Tensor of shape (B, H, W) with class indices (0, 1, 2, ..., num_classes-1)
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        Dice loss (scalar)
    """
    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    
    # Create one-hot encoding for target
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # (B, H, W, C)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    
    # Flatten spatial dimensions
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # (B, C, H*W)
    target_flat = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)  # (B, C, H*W)
    
    # Compute Dice coefficient per class
    intersection = torch.sum(probs_flat * target_flat, dim=2)  # (B, C)
    sum_probs = torch.sum(probs_flat, dim=2)  # (B, C)
    sum_target = torch.sum(target_flat, dim=2)  # (B, C)
    
    # Compute Dice coefficient per class
    dice_coeff = (2. * intersection + smooth) / (sum_probs + sum_target + smooth)  # (B, C)
    
    # Compute Dice loss per class (1 - dice_coeff)
    dice_loss_per_class = 1. - dice_coeff  # (B, C)
    
    if class_weights is None:
        # Average over classes and batch
        return torch.mean(dice_loss_per_class)

    # Weighted average over classes, then mean over batch
    weights = class_weights.to(dice_loss_per_class.device).float()
    weights = weights / (weights.sum() + smooth)
    per_class = dice_loss_per_class.mean(dim=0)  # (C,)
    return torch.sum(per_class * weights)


def focal_loss_multiclass(logits, target, num_classes, alpha=0.25, gamma=2, class_weights=None):
    """
    Multiclass Focal loss.
    
    Args:
        logits: Tensor of shape (B, C, H, W) where C is num_classes
        target: Tensor of shape (B, H, W) with class indices (0, 1, 2, ..., num_classes-1)
        num_classes: Number of classes
        alpha: Weighting factor
        gamma: Focusing parameter
    
    Returns:
        Focal loss (scalar)
    """
    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    
    # Create one-hot encoding for target
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # (B, H, W, C)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    
    # Flatten spatial dimensions
    probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # (B, C, H*W)
    target_flat = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)  # (B, C, H*W)
    
    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(logits.view(logits.size(0), logits.size(1), -1), 
                              target.view(target.size(0), -1), 
                              reduction='none')  # (B, H*W)
    
    # Get probability of true class for each pixel
    probs_true_class = torch.sum(probs_flat * target_flat, dim=1)  # (B, H*W)
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - probs_true_class) ** gamma
    
    # Apply focal weight
    focal_loss_per_pixel = ce_loss * focal_weight
    
    if class_weights is not None:
        weights = class_weights.to(target.device).float()
        per_pixel_weights = weights[target]  # (B, H, W)
        focal_loss_per_pixel = focal_loss_per_pixel * per_pixel_weights.view(target.size(0), -1)

    # Apply alpha weighting if specified
    if alpha >= 0:
        # For multiclass, we can use a simple approach: weight by class frequency
        # Or use uniform alpha
        focal_loss_per_pixel = alpha * focal_loss_per_pixel
    
    return focal_loss_per_pixel.mean()


def evaluate_per_class_accuracy(model, args, num_classes, id2label=None, split='test', max_samples=None):
    """
    Evaluate per-class accuracy on test set.
    
    Args:
        model: The trained model
        args: Training arguments
        num_classes: Number of classes
        id2label: Dictionary mapping class IDs to label names
        split: Dataset split to evaluate ('test')
        max_samples: Maximum number of samples to evaluate (None for all)
    
    Returns:
        Dictionary with per-class accuracies and overall accuracy
    """
    from .segmentation_dataset import hm_dataset_multiclass
    
    # Load test dataset
    use_cache = args.cache_dir is not None and args.cache_dir != ""
    cache_low_res = getattr(args, 'low_res', None)
    if use_cache:
        db_test = hm_dataset_multiclass(
            base_dir=args.root_path,
            split=split,
            transform=None,
            id2label=id2label,
            cache_dir=args.cache_dir,
            cache_img_size=args.img_size,
            cache_low_res=cache_low_res,
            cache_read=args.cache_read,
            cache_write=False,
        )
    else:
        db_test = hm_dataset_multiclass(base_dir=args.root_path, split=split, transform=None, id2label=id2label)
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(db_test):
        indices = list(range(max_samples))
        db_test = Subset(db_test, indices)
    
    loader_kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': args.num_workers}
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = False
        loader_kwargs['prefetch_factor'] = max(2, args.prefetch_factor)
    testloader = DataLoader(db_test, **loader_kwargs)
    
    model.eval()
    
    # Initialize counters for each class
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for sampled_batch in tqdm(testloader, desc="Evaluating per-class accuracy"):
            image_batch = sampled_batch['image'].float().cuda()  # Ensure float32
            label_batch = sampled_batch['label'].cuda()
            
            # Get prediction
            outputs, _ = model(image_batch, True, args.img_size)
            
            # Get predicted class indices
            if 'masks' in outputs:
                # Use masks if available (multimask output)
                pred_logits = outputs['masks']  # (B, C, H, W)
            else:
                # Fallback to low_res_logits
                pred_logits = outputs['low_res_logits']  # (B, C, H, W)
            
            # Get predicted class indices
            pred_classes = torch.argmax(F.softmax(pred_logits, dim=1), dim=1)  # (B, H, W)
            
            # Resize prediction to match label size if needed
            if pred_classes.shape[1:] != label_batch.shape[1:]:
                pred_classes = F.interpolate(
                    pred_classes.unsqueeze(1).float(),
                    size=label_batch.shape[1:],
                    mode='nearest'
                ).squeeze(1).long()
            
            # Clamp predictions and labels to valid range
            pred_classes = torch.clamp(pred_classes, 0, num_classes - 1)
            label_batch = torch.clamp(label_batch.long(), 0, num_classes - 1)
            
            # Flatten for comparison
            pred_flat = pred_classes.flatten()
            label_flat = label_batch.flatten()
            
            # Compute per-class accuracy
            for class_id in range(num_classes):
                class_mask = (label_flat == class_id)
                class_total[class_id] += class_mask.sum().item()
                if class_total[class_id] > 0:
                    class_correct[class_id] += ((pred_flat == label_flat) & class_mask).sum().item()
            
            # Overall accuracy
            correct = (pred_flat == label_flat).sum().item()
            total_correct += correct
            total_pixels += pred_flat.numel()
    
    # Compute accuracies
    class_accuracies = {}
    for class_id in range(num_classes):
        if class_total[class_id] > 0:
            acc = class_correct[class_id] / class_total[class_id]
            class_accuracies[class_id] = acc
        else:
            class_accuracies[class_id] = 0.0
    
    overall_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    # Print results
    logging.info("=" * 60)
    logging.info("Per-Class Accuracy on Test Set:")
    logging.info("-" * 60)
    for class_id in range(num_classes):
        class_name = id2label.get(class_id, f"Class {class_id}") if id2label else f"Class {class_id}"
        acc = class_accuracies[class_id]
        total = class_total[class_id]
        logging.info(f"  {class_name} (ID {class_id}): {acc:.4f} ({class_correct[class_id]}/{total} pixels)")
    logging.info("-" * 60)
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_pixels} pixels)")
    logging.info("=" * 60)
    
    model.train()
    
    return {
        'class_accuracies': class_accuracies,
        'overall_accuracy': overall_accuracy,
        'class_totals': class_total
    }


def evaluate_test_loss(model, args, num_classes, low_res, class_weights=None, split='test', max_samples=None):
    """
    Evaluate loss on the test set.
    """
    from .segmentation_dataset import hm_dataset_multiclass

    use_cache = args.cache_dir is not None and args.cache_dir != ""
    cache_low_res = getattr(args, 'low_res', None)
    if use_cache:
        db_test = hm_dataset_multiclass(
            base_dir=args.root_path,
            split=split,
            transform=None,
            id2label=None,
            cache_dir=args.cache_dir,
            cache_img_size=args.img_size,
            cache_low_res=cache_low_res,
            cache_read=args.cache_read,
            cache_write=False,
        )
    else:
        db_test = hm_dataset_multiclass(base_dir=args.root_path, split=split, transform=None, id2label=None)
    if max_samples is not None and max_samples < len(db_test):
        indices = list(range(max_samples))
        db_test = Subset(db_test, indices)

    loader_kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': args.num_workers}
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = False
        loader_kwargs['prefetch_factor'] = max(2, args.prefetch_factor)
    testloader = DataLoader(db_test, **loader_kwargs)

    model.eval()

    total_loss = 0.0
    total_loss_focal = 0.0
    total_loss_dice = 0.0
    total_loss_focal_hint = 0.0
    total_loss_dice_hint = 0.0
    num_batches = 0

    with torch.no_grad():
        for sampled_batch in tqdm(testloader, desc="Evaluating test loss"):
            image_batch = sampled_batch['image'].float().cuda()
            label_batch = sampled_batch['label'].long().cuda()

            # Resize to training resolution if needed
            if image_batch.shape[2:] != (args.img_size, args.img_size):
                image_batch = F.interpolate(
                    image_batch,
                    size=(args.img_size, args.img_size),
                    mode='bilinear',
                    align_corners=False,
                )
            if label_batch.shape[1:] != (args.img_size, args.img_size):
                label_batch = F.interpolate(
                    label_batch.unsqueeze(1).float(),
                    size=(args.img_size, args.img_size),
                    mode='nearest',
                ).squeeze(1).long()

            low_res_label = F.interpolate(
                label_batch.unsqueeze(1).float(),
                size=(low_res, low_res),
                mode='nearest',
            ).squeeze(1).long()

            outputs, coarse_mask = model(image_batch, True, args.img_size)

            dice_weight = args.dice_param
            input_logits = outputs['low_res_logits']  # (B, C, H, W) for multiclass
            target = torch.clamp(low_res_label, 0, num_classes - 1)

            # coarse_mask is now (B, num_classes, H, W) from the multiclass auto-prompter
            loss_focal = focal_loss_multiclass(input_logits, target, num_classes, class_weights=class_weights)
            loss_dice = dice_loss_multiclass(input_logits, target, num_classes, class_weights=class_weights)
            loss_focal_hint = focal_loss_multiclass(coarse_mask, target, num_classes, class_weights=class_weights)
            loss_dice_hint = dice_loss_multiclass(coarse_mask, target, num_classes, class_weights=class_weights)

            hint_weight = getattr(args, 'hint_loss_weight', 1.0)
            loss = (
                (1 - dice_weight) * loss_focal
                + dice_weight * loss_dice
                + hint_weight * ((1 - dice_weight) * loss_focal_hint + dice_weight * loss_dice_hint)
            )

            total_loss += loss.item()
            total_loss_focal += loss_focal.item()
            total_loss_dice += loss_dice.item()
            total_loss_focal_hint += loss_focal_hint.item()
            total_loss_dice_hint += loss_dice_hint.item()
            num_batches += 1

    model.train()

    if num_batches == 0:
        return {
            'loss': 0.0,
            'loss_focal': 0.0,
            'loss_dice': 0.0,
            'loss_focal_hint': 0.0,
            'loss_dice_hint': 0.0,
        }

    return {
        'loss': total_loss / num_batches,
        'loss_focal': total_loss_focal / num_batches,
        'loss_dice': total_loss_dice / num_batches,
        'loss_focal_hint': total_loss_focal_hint / num_batches,
        'loss_dice_hint': total_loss_dice_hint / num_batches,
    }


def log_label_sanity(dataset, name, max_samples=5):
    counts = {}
    checked = min(max_samples, len(dataset))
    for idx in range(checked):
        sample = dataset[idx]
        label = sample['label']
        if isinstance(label, torch.Tensor):
            label = label.cpu()
        unique, nums = torch.unique(torch.as_tensor(label), return_counts=True)
        for class_id, count in zip(unique.tolist(), nums.tolist()):
            counts[int(class_id)] = counts.get(int(class_id), 0) + int(count)
    logging.info(f"{name} label sanity over {checked} samples: {counts}")


def trainer_hm(args, model, snapshot_path, multimask_output, low_res):
    from .segmentation_dataset import hm_dataset_multiclass, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Load id2label if provided
    id2label = None
    if hasattr(args, 'id2label') and args.id2label:
        id2label_path = Path(args.id2label)
        if id2label_path.exists():
            with open(id2label_path, 'r') as f:
                id2label_data = json.load(f)
                if isinstance(id2label_data, dict):
                    id2label = {int(k): v for k, v in id2label_data.items()}
                else:
                    id2label = {i: str(v) for i, v in enumerate(id2label_data)}
            logging.info(f"Loaded id2label mapping: {id2label}")
    
    use_cache = args.cache_dir is not None and args.cache_dir != ""
    if use_cache:
        db_train = hm_dataset_multiclass(
            base_dir=args.root_path,
            split="train",
            transform=None,
            id2label=id2label,
            cache_dir=args.cache_dir,
            cache_img_size=args.img_size,
            cache_low_res=low_res,
            cache_read=args.cache_read,
            cache_write=args.cache_write,
        )
    else:
        db_train = hm_dataset_multiclass(
            base_dir=args.root_path,
            split="train",
            transform=transforms.Compose(
                [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res], do_random=not args.gpu_aug)]
            ),
            id2label=id2label,
        )
    print("The length of train set is: {}".format(len(db_train)))
    log_label_sanity(db_train, "train")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'worker_init_fn': worker_init_fn,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    trainloader = DataLoader(db_train, **loader_kwargs)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    class_weights = None
    if hasattr(args, 'class_weights') and args.class_weights is not None:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).cuda()

    # Initialize JSON log for epoch losses
    json_log_path = os.path.join(snapshot_path, 'losses.json')
    epoch_logs = []

    for epoch_num in iterator:
        # Accumulate losses for this epoch
        epoch_losses = {'loss': 0.0, 'loss_focal': 0.0, 'loss_dice': 0.0, 'loss_focal_hint': 0.0, 'loss_dice_hint': 0.0}
        num_batches = 0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            if 'low_res_label' in sampled_batch:
                low_res_label_batch = sampled_batch['low_res_label']
            else:
                # Fallback if cache is disabled and transform did not provide low_res_label
                raise ValueError("low_res_label missing from batch; enable cache or RandomGenerator.")
            image_batch = image_batch.float().cuda()
            label_batch = label_batch.long().cuda()
            low_res_label_batch = low_res_label_batch.long().cuda()
            if args.gpu_aug:
                image_batch, label_batch, low_res_label_batch = gpu_random_rot_flip(
                    image_batch, label_batch, low_res_label_batch
                )
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs, coarse_mask = model(image_batch, multimask_output, args.img_size)

            dice_weight = args.dice_param

            # For multiclass, logits are (B, C, H, W) format
            input_logits = outputs['low_res_logits']  # (B, C, H, W) for multiclass
            target = low_res_label_batch.long()  # (B, H, W) with class indices
            
            # Ensure target is in valid range
            target = torch.clamp(target, 0, num_classes - 1)

            loss_focal = focal_loss_multiclass(input_logits, target, num_classes, class_weights=class_weights)
            loss_dice = dice_loss_multiclass(input_logits, target, num_classes, class_weights=class_weights)
            
            # coarse_mask is now (B, num_classes, H, W) from the multiclass auto-prompter
            loss_focal_hint = focal_loss_multiclass(coarse_mask, target, num_classes, class_weights=class_weights)
            loss_dice_hint = dice_loss_multiclass(coarse_mask, target, num_classes, class_weights=class_weights)

            hint_weight = getattr(args, 'hint_loss_weight', 1.0)
            loss = (
                (1 - dice_weight) * loss_focal
                + dice_weight * loss_dice
                + hint_weight * ((1 - dice_weight) * loss_focal_hint + dice_weight * loss_dice_hint)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal_hint', loss_focal_hint, iter_num)
            writer.add_scalar('info/loss_dice_hint', loss_dice_hint, iter_num)

            # Accumulate losses for epoch summary
            epoch_losses['loss'] += loss.item()
            epoch_losses['loss_focal'] += loss_focal.item()
            epoch_losses['loss_dice'] += loss_dice.item()
            epoch_losses['loss_focal_hint'] += loss_focal_hint.item()
            epoch_losses['loss_dice_hint'] += loss_dice_hint.item()
            num_batches += 1

            if iter_num % 20 == 0:
                if image_batch.shape[0] < 2:
                    tqdm.write("Skipping tensorboard")
                else:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    
                    # For multiclass visualization
                    output_masks = outputs['masks']
                    # Get predicted class indices
                    pred_classes = torch.argmax(F.softmax(output_masks, dim=1), dim=1, keepdim=True)
                    # Scale for visualization (use vis_scale to make classes more visible)
                    vis_scale = 255.0 / (num_classes - 1) if num_classes > 1 else 255.0
                    writer.add_image('train/Prediction', pred_classes[1, ...].float() * vis_scale, iter_num)
                    
                    # Ground truth visualization
                    labs = label_batch[1, ...].unsqueeze(0).float() * vis_scale
                    writer.add_image('train/GroundTruth', labs, iter_num)

        # Print average losses at end of epoch
        avg_loss = epoch_losses['loss'] / num_batches
        avg_loss_focal = epoch_losses['loss_focal'] / num_batches
        avg_loss_dice = epoch_losses['loss_dice'] / num_batches
        avg_loss_focal_hint = epoch_losses['loss_focal_hint'] / num_batches
        avg_loss_dice_hint = epoch_losses['loss_dice_hint'] / num_batches
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch_num + 1}/{max_epoch} Training Losses:")
        print(f"  Total Loss:      {avg_loss:.6f}")
        print(f"  Focal Loss:      {avg_loss_focal:.6f}")
        print(f"  Dice Loss:       {avg_loss_dice:.6f}")
        print(f"  Focal Hint Loss: {avg_loss_focal_hint:.6f}")
        print(f"  Dice Hint Loss:  {avg_loss_dice_hint:.6f}")
        
        logging.info('Epoch %d : loss : %f, loss_focal: %f, loss_dice: %f, loss_focal_hint: %f, loss_dice_hint: %f' % 
                     (epoch_num + 1, avg_loss, avg_loss_focal, avg_loss_dice, avg_loss_focal_hint, avg_loss_dice_hint))

        # Log to JSON file (train losses logged every epoch)
        epoch_log_entry = {
            'epoch': epoch_num + 1,
            'train': {
                'loss': avg_loss,
                'loss_focal': avg_loss_focal,
                'loss_dice': avg_loss_dice,
                'loss_focal_hint': avg_loss_focal_hint,
                'loss_dice_hint': avg_loss_dice_hint,
            },
            'learning_rate': lr_,
        }

        # Evaluate test loss every N epochs (default 5) or on final epoch
        test_eval_interval = getattr(args, 'test_eval_interval', 5)
        is_final_epoch = (epoch_num >= max_epoch - 1) or (epoch_num >= stop_epoch - 1)
        if (epoch_num + 1) % test_eval_interval == 0 or is_final_epoch:
            test_subset_size = getattr(args, 'test_subset_size', None)
            acc_results = evaluate_per_class_accuracy(model, args, num_classes, id2label=id2label, split='test', max_samples=test_subset_size)

            test_losses = evaluate_test_loss(
                model,
                args,
                num_classes,
                low_res,
                class_weights=class_weights,
                split='test',
                max_samples=test_subset_size,
            )
            print(f"\nEpoch {epoch_num + 1}/{max_epoch} Test Losses:")
            print(f"  Total Loss:      {test_losses['loss']:.6f}")
            print(f"  Focal Loss:      {test_losses['loss_focal']:.6f}")
            print(f"  Dice Loss:       {test_losses['loss_dice']:.6f}")
            print(f"  Focal Hint Loss: {test_losses['loss_focal_hint']:.6f}")
            print(f"  Dice Hint Loss:  {test_losses['loss_dice_hint']:.6f}")
            print(f"{'='*60}\n")
            
            logging.info(
                'Test loss (epoch %d): loss: %f, loss_focal: %f, loss_dice: %f, loss_focal_hint: %f, loss_dice_hint: %f',
                epoch_num + 1,
                test_losses['loss'],
                test_losses['loss_focal'],
                test_losses['loss_dice'],
                test_losses['loss_focal_hint'],
                test_losses['loss_dice_hint'],
            )
            writer.add_scalar('test/total_loss', test_losses['loss'], epoch_num + 1)
            writer.add_scalar('test/loss_focal', test_losses['loss_focal'], epoch_num + 1)
            writer.add_scalar('test/loss_dice', test_losses['loss_dice'], epoch_num + 1)
            writer.add_scalar('test/loss_focal_hint', test_losses['loss_focal_hint'], epoch_num + 1)
            writer.add_scalar('test/loss_dice_hint', test_losses['loss_dice_hint'], epoch_num + 1)

            # Add test metrics to log entry
            epoch_log_entry['test'] = {
                'loss': test_losses['loss'],
                'loss_focal': test_losses['loss_focal'],
                'loss_dice': test_losses['loss_dice'],
                'loss_focal_hint': test_losses['loss_focal_hint'],
                'loss_dice_hint': test_losses['loss_dice_hint'],
            }
            epoch_log_entry['accuracy'] = {
                'overall': acc_results['overall_accuracy'],
                'per_class': {str(k): v for k, v in acc_results['class_accuracies'].items()},
            }
        else:
            print(f"{'='*60}\n")

        epoch_logs.append(epoch_log_entry)
        
        # Write JSON after each epoch (overwrites to keep it atomic)
        with open(json_log_path, 'w') as f:
            json.dump(epoch_logs, f, indent=2)

        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_dora_parameters(save_mode_path)
            except:
                model.module.save_dora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_dora_parameters(save_mode_path)
            except:
                model.module.save_dora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
