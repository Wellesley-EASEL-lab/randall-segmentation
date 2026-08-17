#!/usr/bin/env python3
"""Train a DINOv3 classifier for the include/exclude review labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

DISTRIBUTED_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_TP",
)


def force_single_process_by_default() -> None:
    """Avoid accidental NCCL initialization from inherited launcher env vars."""
    if "--allow-distributed" in sys.argv:
        return

    for name in DISTRIBUTED_ENV_VARS:
        os.environ.pop(name, None)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


force_single_process_by_default()

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import Trainer, TrainingArguments, set_seed

from randall_pipeline.classifier_data import IncludeCollator, load_datasets, make_processor
from randall_pipeline.classifier_model import DINOv3ForIncludeClassification, save_model_config


DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


class IncludeTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        head_lr: float,
        backbone_lr: float,
        eval_collator: IncludeCollator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset: Any | None = None) -> Any:
        original_collator = self.data_collator
        if self.eval_collator is not None:
            self.data_collator = self.eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    def get_test_dataloader(self, test_dataset: Any) -> Any:
        original_collator = self.data_collator
        if self.eval_collator is not None:
            self.data_collator = self.eval_collator
        try:
            return super().get_test_dataloader(test_dataset)
        finally:
            self.data_collator = original_collator

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            backbone_params = [
                parameter
                for parameter in self.model.backbone.parameters()
                if parameter.requires_grad
            ]
            classifier_params = [
                parameter
                for parameter in self.model.classifier.parameters()
                if parameter.requires_grad
            ]
            parameter_groups = [{"params": classifier_params, "lr": self.head_lr}]
            if backbone_params:
                parameter_groups.append({"params": backbone_params, "lr": self.backbone_lr})
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the saved DatasetDict from build_classifier_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dinov3-vits16-probe"),
        help="Output directory for checkpoints, logs, and metrics.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="DINOv3 model name.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=10.0, help="Linear-probe epochs.")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="After the linear probe, unfreeze DINOv3 and run a fine-tuning phase.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip the frozen-backbone probe and fine-tune from the initial pretrained model.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=float,
        default=5.0,
        help="Additional epochs when --fine-tune is enabled.",
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "patch-mean", "cls-patch-mean"],
        default="cls",
        help="How to pool DINOv3 tokens before classification.",
    )
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps used for scheduler warmup.",
    )
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.2,
        help="Training-only random brightness jitter amount; 0 disables it.",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=0.2,
        help="Training-only random contrast jitter amount; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights.",
    )
    parser.add_argument(
        "--allow-distributed",
        action="store_true",
        help="Honor torchrun/accelerate distributed env vars instead of forcing single-process training.",
    )
    return parser.parse_args()


def compute_class_weights(labels: list[int]) -> list[float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if (counts == 0).any():
        raise ValueError(f"Both classes must be present in train split; got counts {counts}")
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return weights.astype(np.float32).tolist()


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    try:
        metrics["auroc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        metrics["auroc"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


def make_training_args(
    output_dir: Path,
    args: argparse.Namespace,
    epochs: float,
    phase_name: str,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=epochs,
        dataloader_num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name=phase_name,
        seed=args.seed,
    )


def train_phase(
    *,
    model: DINOv3ForIncludeClassification,
    datasets: Any,
    train_collator: IncludeCollator,
    eval_collator: IncludeCollator,
    args: argparse.Namespace,
    phase_name: str,
    epochs: float,
) -> IncludeTrainer:
    trainer = IncludeTrainer(
        model=model,
        args=make_training_args(args.output / phase_name, args, epochs, phase_name),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        eval_collator=eval_collator,
    )
    trainer.train()
    return trainer


def write_metrics(output_dir: Path, trainer: Trainer, final_metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "final_validation_metrics": final_metrics,
        "log_history": trainer.state.log_history,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.brightness_jitter < 0 or args.contrast_jitter < 0:
        raise ValueError("Color jitter amounts must be non-negative")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.skip_probe and not args.fine_tune:
        raise ValueError("--skip-probe requires --fine-tune")

    datasets = load_datasets(args.dataset)
    processor = make_processor(args.model_name, args.image_size)
    train_collator = IncludeCollator(
        processor,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
    )
    eval_collator = IncludeCollator(processor)
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(datasets["train"]["label"])

    model = DINOv3ForIncludeClassification(
        model_name=args.model_name,
        freeze_backbone=not args.skip_probe,
        class_weights=class_weights,
        pooling=args.pooling,
    )

    final_trainer: IncludeTrainer | None = None
    if not args.skip_probe:
        final_trainer = train_phase(
            model=model,
            datasets=datasets,
            train_collator=train_collator,
            eval_collator=eval_collator,
            args=args,
            phase_name="probe",
            epochs=args.epochs,
        )

    if args.fine_tune:
        model.set_backbone_trainable(True)
        final_trainer = train_phase(
            model=model,
            datasets=datasets,
            train_collator=train_collator,
            eval_collator=eval_collator,
            args=args,
            phase_name="fine-tune",
            epochs=args.fine_tune_epochs,
        )

    if final_trainer is None:
        raise RuntimeError("No training phase ran")

    final_metrics = final_trainer.evaluate()
    best_dir = args.output / "checkpoint-best"
    final_trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    save_model_config(model, best_dir)
    write_metrics(args.output, final_trainer, final_metrics)
    print(json.dumps(final_metrics, indent=2))
    print(f"Saved best model to {best_dir}")


if __name__ == "__main__":
    main()
