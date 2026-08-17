# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import cv2
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from .common import LayerNorm2d

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


def point_selection(mask, topk=1):
    """Binary point selection for single-channel masks."""
    # Top-1 point selection
    w, h = mask.shape
    topk_xy = mask.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = torch.tensor([1] * topk, device=mask.device)

    # Top-last point selection
    last_xy = mask.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = torch.tensor([0] * topk, device=mask.device)

    return topk_xy, topk_label, last_xy, last_label


def point_selection_multiclass(mask, num_classes, topk=1):
    """
    Select top-k points per class for multi-class segmentation.
    
    Args:
        mask: (C, H, W) tensor with class logits
        num_classes: number of classes
        topk: number of points to select per class
    
    Returns:
        all_points: (N, 2) tensor of point coordinates (y, x)
        all_labels: (N,) tensor of point labels (1 for foreground, 0 for background)
    """
    # Use softmax probabilities
    probs = F.softmax(mask, dim=0)  # (C, H, W)
    C, H, W = mask.shape
    
    all_points = []
    all_labels = []
    
    # For each non-background class, select top-k points as foreground
    for c in range(1, num_classes):  # Skip background (class 0)
        class_probs = probs[c]  # (H, W)
        topk_vals, topk_idx = class_probs.flatten().topk(topk)
        topk_x = topk_idx // W
        topk_y = topk_idx % W
        points = torch.stack([topk_y, topk_x], dim=1)  # (topk, 2) - format is (y, x)
        all_points.append(points)
        all_labels.append(torch.ones(topk, device=mask.device))  # foreground label = 1
    
    # Select background points from class 0
    bg_probs = probs[0]  # (H, W)
    topk_vals, topk_idx = bg_probs.flatten().topk(topk)
    topk_x = topk_idx // W
    topk_y = topk_idx % W
    bg_points = torch.stack([topk_y, topk_x], dim=1)  # (topk, 2)
    all_points.append(bg_points)
    all_labels.append(torch.zeros(topk, device=mask.device))  # background label = 0
    
    return torch.cat(all_points, dim=0), torch.cat(all_labels, dim=0)

class SPGen(nn.Module):
    VALID_VARIANTS = ("original_head", "depthwise")

    def __init__(self, num_classes=1, variant="depthwise"):
        super(SPGen, self).__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown SPGen variant '{variant}'. Expected one of {self.VALID_VARIANTS}.")
        self.num_classes = num_classes
        self.variant = variant

        if variant == "original_head":
            self.hint = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, stride=1),
                LayerNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 16, kernel_size=1, stride=1),
                LayerNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, num_classes, kernel_size=1),
            )
        else:
            self.hint = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
                LayerNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 64, kernel_size=1, stride=1),
                LayerNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 16, kernel_size=1, stride=1),
                LayerNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, num_classes, kernel_size=1),
            )

    def forward(self, x):
        x = self.hint(x)
        return x

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        num_classes: int = 1,
        spgen_variant: str = "depthwise",
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
          num_classes (int): Number of output classes for multi-class segmentation.
          spgen_variant (str): SPGen head variant, "depthwise" or "original_head".
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.num_classes = num_classes
        self.spgen_variant = spgen_variant
        self.spgen = SPGen(num_classes, variant=spgen_variant)

        self.spgen1 = SPGen(num_classes, variant=spgen_variant)
        self.spgen2 = SPGen(num_classes, variant=spgen_variant)
        self.spgen3 = SPGen(num_classes, variant=spgen_variant)

        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)
        image_embeddings, stored_states = self.image_encoder(input_images)

        coarse_mask0 = self.spgen(image_embeddings)
        coarse_mask1 = self.spgen1(stored_states[0])
        coarse_mask2 = self.spgen2(stored_states[1])
        coarse_mask3 = self.spgen3(stored_states[2])
        # Average across the 4 SPGen outputs: shape is (B, num_classes, H, W)
        coarse_mask = torch.mean(torch.stack([coarse_mask0, coarse_mask1, coarse_mask2, coarse_mask3], dim=0), dim=0)

        coarse_mask_up4 = F.interpolate(coarse_mask, size=(int(image_size/4), int(image_size/4)), mode="bilinear")
        coarse_mask_up16 = F.interpolate(coarse_mask, size=(image_size, image_size), mode="bilinear")

        # Compute probabilities differently based on num_classes
        if self.num_classes > 1:
            # Multiclass: use softmax
            spgen_prob = F.softmax(coarse_mask.detach(), dim=1)  # (B, C, H, W)
        else:
            # Binary: use sigmoid
            spgen_prob = torch.sigmoid(coarse_mask.detach())  # (B, 1, H, W)

        outputs = {
            'masks': [],
            'iou_predictions': [],
            'low_res_logits': []
        }

        for idx in range(batched_input.shape[0]):  # for each batch

            if self.num_classes > 1:
                # Multiclass: create attention mask based on combined foreground
                # Get predicted class per pixel (argmax across classes)
                pred_classes = torch.argmax(coarse_mask[idx], dim=0)  # (H, W)
                # Foreground is where predicted class != 0 (background)
                fg_mask = (pred_classes != 0).float()  # (H, W)
                # Attention mask: True where we want to mask (background regions)
                attn_mask = (fg_mask.unsqueeze(0).unsqueeze(0).flatten(2) < 0.5).bool()  # (1, 1, H*W)
            else:
                # Binary: use sigmoid threshold
                attn_mask = (coarse_mask[idx].sigmoid().unsqueeze(0).unsqueeze(0).flatten(3) < 0.5).bool()
            
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, -1e9)
            attn_mask = new_attn_mask
            attn_mask = attn_mask.detach()

            # Positive-negative location prior
            if self.num_classes > 1:
                # Multiclass: use per-class point selection
                topk_xy, topk_label = point_selection_multiclass(
                    coarse_mask_up16[idx], self.num_classes, topk=2
                )
                fg_points = (topk_xy.unsqueeze(0), topk_label.unsqueeze(0))
            else:
                # Binary: use original point selection
                topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(
                    coarse_mask_up16[idx].squeeze(0), topk=2
                )
                topk_xy = torch.cat([topk_xy_i, last_xy_i], dim=0)
                topk_label = torch.cat([topk_label_i, last_label_i], dim=0)
                fg_points = (topk_xy.unsqueeze(0), topk_label.unsqueeze(0))

            # Get target semantic embedding
            image_feat = image_embeddings[idx].permute(1, 2, 0).detach()  # (H, W, C)
            
            if self.num_classes > 1:
                # Multiclass: extract features from all non-background regions
                # Use combined foreground probability (1 - background probability)
                fg_prob = 1.0 - spgen_prob[idx, 0]  # (H, W) - probability of being any foreground class
                target_feat = image_feat[fg_prob > 0.5]
            else:
                # Binary: use original threshold
                target_feat = image_feat[spgen_prob[idx].squeeze() > 0.5]
            
            if target_feat.numel() > 0:
                target_embedding = target_feat.mean(0).unsqueeze(0).unsqueeze(0)
            else:
                target_embedding = None

            if target_embedding is not None and torch.isnan(target_embedding).any():
                target_embedding = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=fg_points, boxes=None, masks=None
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                attn_mask=attn_mask,
                target_embedding=target_embedding
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=(image_size, image_size),
                original_size=(image_size, image_size)
            )
            outputs['masks'].append(masks)
            outputs['iou_predictions'].append(iou_predictions)
            outputs['low_res_logits'].append(low_res_masks)

        outputs['masks'] = torch.stack(outputs['masks'], dim=0).squeeze(1)
        outputs['iou_predictions'] = torch.stack(outputs['iou_predictions'], dim=0).squeeze(1)
        outputs['low_res_logits'] = torch.stack(outputs['low_res_logits'], dim=0).squeeze(1)

        return outputs, coarse_mask_up4

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
