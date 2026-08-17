"""DINOv3 binary classifier for include/exclude labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DINOv3ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput


MODEL_CONFIG_NAME = "model_config.json"
PoolingMode = Literal["cls", "patch-mean", "cls-patch-mean"]


class DINOv3ForIncludeClassification(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        freeze_backbone: bool = True,
        class_weights: list[float] | torch.Tensor | None = None,
        pooling: PoolingMode = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "patch-mean", "cls-patch-mean"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.model_name = model_name
        self.num_labels = num_labels
        self.pooling = pooling
        self.backbone = DINOv3ViTModel.from_pretrained(model_name)
        pooled_size = self.backbone.config.hidden_size
        if pooling == "cls-patch-mean":
            pooled_size *= 2
        self.classifier = nn.Linear(pooled_size, num_labels)
        self.set_backbone_trainable(not freeze_backbone)

        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float)
        else:
            weights = torch.ones(num_labels, dtype=torch.float)
        self.register_buffer("class_weights", weights)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0]
        if self.pooling == "cls":
            return cls_token

        num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        patch_start = 1 + int(num_register_tokens)
        patch_tokens = hidden_states[:, patch_start:]
        if patch_tokens.shape[1] == 0:
            patch_tokens = hidden_states[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)

        if self.pooling == "patch-mean":
            return patch_mean
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = self.pool_hidden_states(outputs.last_hidden_state)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the exact pooled representation consumed by the classifier head."""
        outputs = self.backbone(pixel_values=pixel_values)
        return self.pool_hidden_states(outputs.last_hidden_state)

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "pooling": self.pooling,
            "class_weights": self.class_weights.detach().cpu().tolist(),
        }


def save_model_config(model: DINOv3ForIncludeClassification, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / MODEL_CONFIG_NAME).write_text(
        json.dumps(model.config_dict(), indent=2) + "\n", encoding="utf-8"
    )


def load_model_config(checkpoint_dir: str | Path) -> dict[str, Any]:
    config_path = Path(checkpoint_dir) / MODEL_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {config_path}. Use a checkpoint written by train_classifier.py."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_classifier(checkpoint_dir: str | Path, map_location: str | torch.device = "cpu"):
    checkpoint_path = Path(checkpoint_dir)
    config = load_model_config(checkpoint_path)
    model = DINOv3ForIncludeClassification(
        model_name=config["model_name"],
        num_labels=config.get("num_labels", 2),
        freeze_backbone=False,
        class_weights=config.get("class_weights"),
        pooling=config.get("pooling", "cls"),
    )

    state_dict_path = checkpoint_path / "pytorch_model.bin"
    safetensors_path = checkpoint_path / "model.safetensors"
    if state_dict_path.exists():
        state_dict = torch.load(state_dict_path, map_location=map_location)
    elif safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device=str(map_location))
    else:
        raise FileNotFoundError(
            f"Could not find pytorch_model.bin or model.safetensors in {checkpoint_path}"
        )

    model.load_state_dict(state_dict)
    return model
