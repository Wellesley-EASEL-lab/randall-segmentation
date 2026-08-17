# Suppress OpenCV TIFF warnings
import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from randall_pipeline.constants import SEGMENTATION_MODEL
from randall_pipeline.inference import validate_adapter_config
from randall_pipeline.io import stable_id
from randall_pipeline.segmentation_model import DoRA_Sam

# Suppress PIL decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None


def config_to_dict(config):
    """Parse config file to dictionary with proper type conversion."""
    config_path = Path(config)
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        validate_adapter_config(payload)
        return {
            "vit_name": "vit_l",
            "num_classes": int(payload["num_classes"]),
            "img_size": int(payload["preprocessing"]["image_size"]),
            "rank": int(payload["adapter"]["rank"]),
            "spgen_variant": str(payload["adapter"]["spgen_variant"]),
        }
    items_dict = {}
    
    # Define type mappings for arguments that need conversion
    int_args = {'num_classes', 'img_size', 'input_size', 'seed', 'deterministic', 'rank', 'max_iterations', 'max_epochs', 'stop_epoch', 'batch_size', 'n_gpu'}
    float_args = {'base_lr', 'dice_param'}
    bool_args = {'warmup', 'AdamW', 'is_pretrain'}
    
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        line = items[i].strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(': ', 1)
        
        # Handle "None" string
        if value == "None":
            items_dict[key] = None
        # Convert value to appropriate type
        elif key in int_args:
            items_dict[key] = int(value)
        elif key in float_args:
            items_dict[key] = float(value)
        elif key in bool_args:
            items_dict[key] = value.lower() in ('true', '1', 'yes')
        else:
            items_dict[key] = value
    
    return items_dict


def pad_to_square(img: Image.Image) -> tuple[Image.Image, dict]:
    """
    Pad image to square shape by adding padding to the shorter dimension.
    
    Returns:
        padded_img: Padded PIL Image
        padding_info: Dict with pad_left, pad_right, pad_top, pad_bottom
    """
    width, height = img.size
    
    # Calculate padding needed to make square
    if width > height:
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
        pad_left = 0
        pad_right = 0
        new_size = width
    elif height > width:
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
        pad_top = 0
        pad_bottom = 0
        new_size = height
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0
        new_size = width
    
    # Create padded image
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        padded_img = Image.new(img.mode, (new_size, new_size), 0)
        padded_img.paste(img, (pad_left, pad_top))
    else:
        padded_img = img.copy()
    
    padding_info = {
        'pad_left': pad_left,
        'pad_right': pad_right,
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
    }
    
    return padded_img, padding_info


def tile_image(img: Image.Image) -> list[tuple[Image.Image, str]]:
    """
    Tile image into 6 equal-sized tiles (3 rows x 2 columns).
    
    Returns:
        List of tuples: (tile_img, tile_suffix)
        Tile suffixes: '_tile_0', '_tile_1', ..., '_tile_5'
        Tiles are ordered: row 0 cols 0-1, row 1 cols 0-1, row 2 cols 0-1
    """
    width, height = img.size
    tile_h = height // 3
    tile_w = width // 2
    
    tiles = []
    tile_idx = 0
    
    for row in range(3):
        for col in range(2):
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w if col < 1 else width  # Last column gets remainder
            bottom = top + tile_h if row < 2 else height  # Last row gets remainder
            
            tile_img = img.crop((left, top, right, bottom))
            tile_suffix = f'_tile_{tile_idx}'
            tiles.append((tile_img, tile_suffix))
            tile_idx += 1
    
    return tiles


def load_image(path: Path) -> Image.Image:
    """Load image from file (JPEG, PNG, or TIFF)."""
    try:
        # Use context manager to ensure the underlying file handle is closed.
        # PIL can otherwise keep many files open when processing large folders.
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img.copy()
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def preprocess_image(img: Image.Image, img_size: int, tile_threshold: int = 1024) -> list[tuple[Image.Image, dict, str]]:
    """
    Preprocess image: tile if needed, pad to square, resize.
    
    Returns:
        List of tuples: (processed_img, padding_info, tile_suffix)
        For single images: one tuple with tile_suffix=''
        For tiled images: 6 tuples with tile_suffix='_tile_0', '_tile_1', etc.
    """
    original_width, original_height = img.size
    results = []
    
    # Check if tiling is needed
    if original_width > tile_threshold or original_height > tile_threshold:
        # Tile the image
        tiles = tile_image(img)
        for tile_img, tile_suffix in tiles:
            # Pad to square
            padded_img, padding_info = pad_to_square(tile_img)
            # Resize to model input size
            resized_img = padded_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            results.append((resized_img, padding_info, tile_suffix))
    else:
        # Single image - pad to square and resize
        padded_img, padding_info = pad_to_square(img)
        resized_img = padded_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        results.append((resized_img, padding_info, ''))
    
    return results


def run_inference(net, image_tensor, img_size: int, multimask_output: bool) -> np.ndarray:
    """
    Run inference on a single image tensor.
    
    Args:
        net: DORA-SAM model
        image_tensor: Tensor of shape (1, C, H, W) normalized to [0, 1]
        img_size: Model input size
        multimask_output: Whether to use multimask output
    
    Returns:
        Prediction array of shape (H, W) with class IDs at img_size resolution
    """
    preds = run_inference_batch(net, image_tensor, img_size, multimask_output)
    if preds.ndim != 3 or preds.shape[0] != 1:
        raise ValueError(f"Expected single prediction of shape (1, H, W), got {preds.shape}")
    return preds[0]


def run_inference_batch(net, image_tensor, img_size: int, multimask_output: bool) -> np.ndarray:
    """
    Run inference on a batch of image tensors.

    Args:
        net: DORA-SAM model
        image_tensor: Tensor of shape (B, C, H, W) normalized to [0, 1]
        img_size: Model input size
        multimask_output: Whether to use multimask output

    Returns:
        Prediction array of shape (B, H, W) with uint8 class IDs at img_size resolution
    """
    if image_tensor.dim() != 4:
        raise ValueError(f"Expected image_tensor to be 4D (B, C, H, W), got shape {tuple(image_tensor.shape)}")

    with torch.no_grad():
        # Ensure image batch is on GPU
        if image_tensor.device.type != 'cuda':
            image_tensor = image_tensor.cuda(non_blocking=True)

        outputs, _coarse_mask = net(image_tensor, multimask_output, img_size)
        output_masks = outputs['masks']

        # output_masks should be (B, C, H, W)
        if output_masks.dim() != 4:
            raise ValueError(f"Expected output_masks to be 4D (B, C, H, W), got shape {tuple(output_masks.shape)}")

        probs = torch.softmax(output_masks, dim=1)
        pred = torch.argmax(probs, dim=1)  # (B, H, W)
        return pred.cpu().detach().numpy().astype(np.uint8)


def parse_tile_position_tiled(filename: str):
    """
    Parse tile position from filename pattern: {name}_tile_{idx}.tif or similar
    
    Returns:
        base_name (str): Base name without tile suffix
        row (int): Row index (0-2 for tiled, 0 for single)
        col (int): Column index (0-1 for tiled, 0 for single)
        tile_idx (int): Tile index (0-5 for tiled, -1 for single)
        is_tiled (bool): True if image is part of a 6-tile set
    """
    # Remove extension
    base_name = filename.rsplit('.', 1)[0]
    
    # Match pattern: _tile_{idx} at the end
    match = re.search(r'_tile_(\d+)$', base_name)
    if match:
        tile_idx = int(match.group(1))
        # Extract base name (everything before _tile_{idx})
        name_end = match.start()
        base_name_only = base_name[:name_end]
        
        # Map tile index to row/col (3 rows x 2 cols)
        row = tile_idx // 2
        col = tile_idx % 2
        
        return base_name_only, row, col, tile_idx, True
    else:
        # Single image (no tiling)
        return base_name, 0, 0, -1, False


def compute_tile_original_size(original_width, original_height, row, col, tile_idx):
    """Compute the original (pre-padding) tile width/height."""
    if tile_idx == -1 or row is None or col is None:
        return original_width, original_height

    tile_h = original_height // 3
    tile_w = original_width // 2
    if row == 2:
        tile_h = original_height - 2 * tile_h
    if col == 1:
        tile_w = original_width - tile_w
    return tile_w, tile_h


def unpad_prediction(prediction: np.ndarray, padding_info: dict, original_width: int, original_height: int, 
                     row: int, col: int, tile_idx: int) -> np.ndarray:
    """
    Remove padding from a prediction tile using metadata.
    
    Args:
        prediction: Prediction array (H, W) with class IDs
        padding_info: Dict with pad_left, pad_right, pad_top, pad_bottom
        original_width: Original full image width
        original_height: Original full image height
        row: Tile row index
        col: Tile column index
        tile_idx: Tile index (-1 for single images)
    
    Returns:
        Unpadded prediction array
    """
    if not padding_info or (padding_info.get('pad_left', 0) == 0 and 
                           padding_info.get('pad_right', 0) == 0 and
                           padding_info.get('pad_top', 0) == 0 and
                           padding_info.get('pad_bottom', 0) == 0):
        return prediction
    
    pred_h, pred_w = prediction.shape
    
    # Compute original tile size
    tile_orig_w, tile_orig_h = compute_tile_original_size(
        original_width, original_height, row, col, tile_idx
    )
    padded_dim = max(tile_orig_w, tile_orig_h)
    if padded_dim <= 0:
        return prediction
    
    # Scale padding values to prediction size
    scale = pred_h / float(padded_dim)
    pad_left = int(round(padding_info.get('pad_left', 0) * scale))
    pad_right = int(round(padding_info.get('pad_right', 0) * scale))
    pad_top = int(round(padding_info.get('pad_top', 0) * scale))
    pad_bottom = int(round(padding_info.get('pad_bottom', 0) * scale))
    
    # Crop out padding
    x_start = max(0, pad_left)
    y_start = max(0, pad_top)
    x_end = max(x_start + 1, pred_w - pad_right)
    y_end = max(y_start + 1, pred_h - pad_bottom)
    x_end = min(x_end, pred_w)
    y_end = min(y_end, pred_h)
    
    unpadded = prediction[y_start:y_end, x_start:x_end]
    return unpadded


def resize_class_map(
    prediction: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize a 2D class-ID map without introducing interpolated class values."""
    if prediction.ndim != 2:
        raise ValueError(f"Expected a 2D prediction, got shape {prediction.shape}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive, got {target_width}x{target_height}"
        )
    if prediction.shape == (target_height, target_width):
        return prediction

    prediction_image = Image.fromarray(prediction.astype(np.uint8, copy=False))
    resized = prediction_image.resize(
        (target_width, target_height), Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8)


def retile_predictions(tiles: list, original_width: int, original_height: int) -> np.ndarray:
    """
    Retile grouped tiles back into full image.
    
    Args:
        tiles: list of dicts with:
            - 'prediction': numpy array (H, W) uint8 class IDs
            - 'row': int
            - 'col': int
            - 'tile_idx': int (-1 for single images)
            - 'padding': dict with padding info
        original_width: Original full image width
        original_height: Original full image height
    
    Returns:
        Retiled prediction array (H, W) with class IDs
    """
    tiles = sorted(tiles, key=lambda t: t['tile_idx'])
    
    # Check if this is a single image or tiled
    if len(tiles) == 1 and tiles[0]['tile_idx'] == -1:
        # Single image - unpad and restore the exact source dimensions.
        tile = tiles[0]
        unpadded = unpad_prediction(
            tile['prediction'],
            tile.get('padding', {}),
            original_width,
            original_height,
            tile['row'],
            tile['col'],
            tile['tile_idx']
        )
        return resize_class_map(unpadded, original_width, original_height)
    
    # Tiled image - reconstruct 3x2 grid
    if len(tiles) != 6:
        raise ValueError(f"Expected 6 tiles, but found {len(tiles)} tiles")
    
    # Unpad all tiles first
    unpadded_tiles = []
    for tile in tiles:
        unpadded = unpad_prediction(
            tile['prediction'],
            tile.get('padding', {}),
            original_width,
            original_height,
            tile['row'],
            tile['col'],
            tile['tile_idx']
        )
        tile_width, tile_height = compute_tile_original_size(
            original_width,
            original_height,
            tile['row'],
            tile['col'],
            tile['tile_idx'],
        )
        unpadded = resize_class_map(unpadded, tile_width, tile_height)
        unpadded_tiles.append({
            **tile,
            'prediction': unpadded
        })
    
    # Determine per-row heights and per-col widths (after de-padding)
    row_heights = [0, 0, 0]
    col_widths = [0, 0]
    for tile in unpadded_tiles:
        tile_h, tile_w = tile['prediction'].shape
        row_heights[tile['row']] = max(row_heights[tile['row']], tile_h)
        col_widths[tile['col']] = max(col_widths[tile['col']], tile_w)
    
    # Create canvas (3 rows x 2 cols)
    canvas_h = sum(row_heights)
    canvas_w = sum(col_widths)
    canvas_pred = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Place each tile in its position
    for tile in unpadded_tiles:
        row = tile['row']
        col = tile['col']
        
        # Calculate position
        y_start = sum(row_heights[:row])
        x_start = sum(col_widths[:col])
        y_end = y_start + tile['prediction'].shape[0]
        x_end = x_start + tile['prediction'].shape[1]
        max_y = y_start + row_heights[row]
        max_x = x_start + col_widths[col]
        y_end = min(y_end, max_y)
        x_end = min(x_end, max_x)
        actual_h = y_end - y_start
        actual_w = x_end - x_start
        
        if actual_h > 0 and actual_w > 0:
            canvas_pred[y_start:y_end, x_start:x_end] = tile['prediction'][:actual_h, :actual_w]
    
    # Each tile has already been restored to its source-space dimensions. Keep
    # this final resize as a guard against any future non-standard tile layout.
    return resize_class_map(canvas_pred, original_width, original_height)


def get_color_palette(num_classes: int) -> list[tuple[int, int, int]]:
    """
    Generate a color palette for multiclass visualization (same as viewer_results_multiclass.py).
    """
    base_colors = [
        (0, 0, 0),        # 0: Black (background)
        (255, 0, 0),      # 1: Red
        (0, 255, 0),      # 2: Green
        (0, 0, 255),      # 3: Blue
        (255, 255, 0),    # 4: Yellow
        (255, 0, 255),    # 5: Magenta
        (0, 255, 255),    # 6: Cyan
        (255, 128, 0),    # 7: Orange
        (128, 0, 255),    # 8: Purple
        (255, 192, 203),  # 9: Pink
        (165, 42, 42),    # 10: Brown
        (128, 128, 128),  # 11: Gray
        (255, 165, 0),    # 12: Orange
        (0, 128, 0),      # 13: Dark Green
        (128, 0, 0),      # 14: Dark Red
        (0, 0, 128),      # 15: Dark Blue
    ]

    if num_classes > len(base_colors):
        import colorsys
        additional_colors = []
        for i in range(len(base_colors), num_classes):
            hue = (i - len(base_colors)) / max(1, num_classes - len(base_colors))
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.2
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            additional_colors.append(tuple(int(c * 255) for c in rgb))
        base_colors.extend(additional_colors)

    return base_colors[:num_classes]


def build_id2label_with_colors(num_classes: int, id2label: dict | None, color_palette: list[tuple[int, int, int]]):
    labels = {}
    for class_id in range(num_classes):
        if id2label and class_id in id2label:
            name = id2label[class_id]
        elif class_id == 0:
            name = "background"
        else:
            name = f"class {class_id}"
        color = color_palette[class_id] if class_id < len(color_palette) else (0, 0, 0)
        labels[str(class_id)] = {
            "label": name,
            "color": list(color),
        }
    return labels


def tile_suffix_to_position(tile_suffix: str) -> tuple[int, int, int]:
    """
    Convert a tile suffix ('' or '_tile_{idx}') to (row, col, tile_idx).

    This avoids parsing the base filename (which may itself contain '_tile_').
    """
    if tile_suffix and tile_suffix.startswith("_tile_"):
        try:
            tile_idx = int(tile_suffix.split("_")[-1])
        except Exception:
            tile_idx = -1
        if tile_idx >= 0:
            return tile_idx // 2, tile_idx % 2, tile_idx
    return 0, 0, -1


def iter_image_tile_lists(image_paths: list[Path], input_root: Path, img_size: int, tile_threshold: int):
    """
    Generator that yields one dict per image, containing a list of tile entries.

    On errors, yields a dict with keys: {"img_path": Path, "error": str}.
    """
    for img_path in image_paths:
        try:
            # Load image (closes file handle immediately)
            img = load_image(img_path)
            original_width, original_height = img.size

            # Preprocess: tile if needed, pad, resize
            processed_tiles = preprocess_image(img, img_size, tile_threshold)

            relative = img_path.resolve().relative_to(input_root.resolve())
            base_name = f"{stable_id(relative.as_posix())}-{img_path.stem}"
            image_key = str(img_path.resolve())

            tile_entries = []
            for processed_img, padding_info, tile_suffix in processed_tiles:
                img_array = np.array(processed_img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                row, col, tile_idx = tile_suffix_to_position(tile_suffix)
                tile_entries.append({
                    "image_key": image_key,
                    "tile_suffix": tile_suffix,
                    "padding_info": padding_info,
                    "img_tensor": img_tensor,
                    "row": row,
                    "col": col,
                    "tile_idx": tile_idx,
                })

            yield {
                "image_key": image_key,
                "img_path": img_path,
                "base_name": base_name,
                "original_width": original_width,
                "original_height": original_height,
                "tile_entries": tile_entries,
            }
        except Exception as e:
            import traceback
            yield {"img_path": img_path, "error": str(e), "traceback": traceback.format_exc()}


def save_segmentation_map(prediction: np.ndarray, output_path: Path, color_palette: list[tuple[int, int, int]]):
    """
    Save segmentation map to file.
    
    Args:
        prediction: Prediction array (H, W) with class IDs
        output_path: Path to save the output
        color_palette: List of RGB tuples for each class
    """
    num_classes = len(color_palette)
    h, w = prediction.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        mask = prediction == class_id
        if np.any(mask):
            colorized[mask] = color_palette[class_id]

    img = Image.fromarray(colorized, mode='RGB')
    img.save(output_path)


class TileBatchInferencer:
    """
    Stream tiles from many images, run GPU inference in batches, then finalize per-image outputs.

    This class exists mostly to keep `main()` readable (no deeply nested helper functions).
    """

    def __init__(
        self,
        *,
        net,
        img_size: int,
        multimask_output: bool,
        infer_batch_size: int,
        output_dir: Path,
        color_palette: list[tuple[int, int, int]],
        jsonl_f,
        pbar,
        input_root: Path,
    ):
        self.net = net
        self.img_size = img_size
        self.multimask_output = multimask_output
        self.infer_batch_size = infer_batch_size
        self.output_dir = output_dir
        self.color_palette = color_palette
        self.jsonl_f = jsonl_f
        self.pbar = pbar
        self.input_root = input_root.resolve()

        self.tile_queue = deque()
        self.pending_images: dict[str, dict] = {}
        self.processed = 0

    def on_image_error(self, item: dict):
        img_path = item.get("img_path")
        logging.error(f"Error processing {img_path}: {item.get('error')}")
        if item.get("traceback"):
            logging.error(item["traceback"])
        self.pbar.update(1)

    def enqueue_image(self, item: dict):
        image_key = item["image_key"]
        self.pending_images[image_key] = {
            "img_path": item["img_path"],
            "base_name": item["base_name"],
            "original_width": item["original_width"],
            "original_height": item["original_height"],
            "expected": len(item["tile_entries"]),
            "done": 0,
            "tiles": [],
        }
        self.tile_queue.extend(item["tile_entries"])

    def drain_full_batches(self):
        while self.infer_batch_size > 0 and len(self.tile_queue) >= self.infer_batch_size:
            tile_batch = [self.tile_queue.popleft() for _ in range(self.infer_batch_size)]
            self.process_tile_batch(tile_batch)

    def flush(self):
        # Flush remaining tiles (may be < infer_batch_size).
        if len(self.tile_queue) > 0:
            tile_batch = list(self.tile_queue)
            self.tile_queue.clear()
            self.process_tile_batch(tile_batch)

        if self.pending_images:
            # This should not happen unless inference failed mid-way.
            logging.error(
                f"Internal error: {len(self.pending_images)} images still have incomplete tiles after final flush; skipping them."
            )
            self.pbar.update(len(self.pending_images))
            self.pending_images.clear()

    def finalize_image(self, image_key: str):
        ctx = self.pending_images.pop(image_key)
        image_tiles = ctx["tiles"]
        retiled_prediction = retile_predictions(
            image_tiles,
            ctx["original_width"],
            ctx["original_height"],
        )

        output_filename = f"{ctx['base_name']}_segmentation.png"
        output_path = self.output_dir / output_filename
        save_segmentation_map(
            retiled_prediction,
            output_path,
            color_palette=self.color_palette,
        )

        record = {
            "asset_id": stable_id(ctx["img_path"].resolve().relative_to(self.input_root).as_posix()),
            "input_image": os.path.relpath(ctx["img_path"].resolve(), self.output_dir.resolve()),
            "segmentation_map": output_path.resolve().relative_to(self.output_dir.resolve()).as_posix(),
            "page": ctx["img_path"].stem,
        }
        self.jsonl_f.write(json.dumps(record) + "\n")
        self.jsonl_f.flush()

        self.processed += 1
        self.pbar.update(1)

        # Optional: encourage release of cached blocks occasionally
        if torch.cuda.is_available() and (self.processed % 25 == 0) and self.processed > 0:
            torch.cuda.empty_cache()

    def process_tile_batch(self, tile_batch: list[dict]):
        batch_tensor = torch.cat([t["img_tensor"] for t in tile_batch], dim=0)
        batch_preds = run_inference_batch(
            self.net, batch_tensor, self.img_size, self.multimask_output
        )

        for t, prediction in zip(tile_batch, batch_preds):
            if prediction.shape != (self.img_size, self.img_size):
                raise ValueError(
                    f"Prediction shape {prediction.shape} doesn't match expected ({self.img_size}, {self.img_size})"
                )
                # prediction = zoom(
                #     prediction,
                #     (self.img_size / prediction.shape[0], self.img_size / prediction.shape[1]),
                #     order=0,
                # ).astype(np.uint8)

            image_key = t["image_key"]
            ctx = self.pending_images.get(image_key)
            if ctx is None:
                raise ValueError(f"image not found")

            ctx["tiles"].append({
                "prediction": prediction,
                "row": t["row"],
                "col": t["col"],
                "tile_idx": t["tile_idx"],
                "padding": t["padding_info"],
            })
            ctx["done"] += 1
            if ctx["done"] == ctx["expected"]:
                self.finalize_image(image_key)

        # Help the GC / CUDA allocator on large batches
        del batch_tensor
        for t in tile_batch:
            del t["img_tensor"]


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on a folder of unlabeled images using a trained DORA-SAM model'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing input images (JPEG, PNG, TIFF)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Folder for output segmentation maps')
    parser.add_argument('--output_jsonl', type=str, default='predictions.jsonl',
                        help='JSONL manifest filename written inside output_dir (default: predictions.jsonl)')
    parser.add_argument('--dora_ckpt', type=str, default=None,
                        help='Local trained DoRA checkpoint; defaults to the hosted Randall adapter')
    parser.add_argument('--model-repo', default=SEGMENTATION_MODEL,
                        help='Hugging Face model repository for adapter/config/labels')
    parser.add_argument('--model-revision', default=None,
                        help='Optional immutable Hugging Face model revision')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Separately obtained Meta SAM ViT-L base checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_l',
                        help='ViT model type (hosted adapter requires vit_l)')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of classes (default: 8)')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Model input size (default: 1024)')
    parser.add_argument('--rank', type=int, default=8,
                        help='DoRA rank (hosted adapter uses 8)')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file from training (overrides defaults)')
    parser.add_argument('--id2label', type=str, default=None,
                        help='Path to id2label.json for colorized output')
    parser.add_argument('--tile_threshold', type=int, default=1024,
                        help='Size threshold for tiling (default: 1024)')
    parser.add_argument('--infer_batch_size', '--inference_batch_size', type=int, default=12,
                        help='Batch size for model forward-pass (tiles per GPU batch; default: 12)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='Whether to use deterministic operations')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively discover images below input_dir')
    parser.add_argument('--spgen_variant', type=str, default='depthwise',
                        choices=['original_head', 'depthwise'],
                        help='SPGen head variant (default: depthwise)')
    
    args = parser.parse_args()

    if args.dora_ckpt is None:
        args.dora_ckpt = hf_hub_download(
            args.model_repo, "model.pth", revision=args.model_revision
        )
        if args.config is None:
            args.config = hf_hub_download(
                args.model_repo, "config.json", revision=args.model_revision
            )
        if args.id2label is None:
            args.id2label = hf_hub_download(
                args.model_repo, "id2label.json", revision=args.model_revision
            )
    
    # Load config file if provided
    if args.config is not None:
        config_dict = config_to_dict(args.config)
        cli_keys = {arg.lstrip("-") for arg in sys.argv if arg.startswith("--")}
        for key in config_dict:
            # Skip keys explicitly provided on CLI
            if key in cli_keys:
                continue
            # Skip None values from config
            if config_dict[key] is None and key == 'dora_ckpt':
                continue
            setattr(args, key, config_dict[key])
    
    # Set up deterministic behavior
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
    
    # Load id2label if provided
    id2label = None
    if args.id2label:
        id2label_path = Path(args.id2label)
        if not id2label_path.exists():
            raise ValueError(f"--id2label file not found (specified on CLI)")

        with open(id2label_path, 'r') as f:
            id2label_data = json.load(f)
            if isinstance(id2label_data, dict):
                id2label = {int(k): v for k, v in id2label_data.items()}
            else:
                id2label = {i: str(v) for i, v in enumerate(id2label_data)}
        logging.info(f"Loaded id2label: {id2label}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / args.output_jsonl
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info(f"Starting inference with args: {args}")
    
    # Load model
    logging.info("Loading SAM model...")
    sam, _img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
        spgen_variant=args.spgen_variant,
    )
    
    logging.info("Loading DORA parameters...")
    if not torch.cuda.is_available():
        raise RuntimeError("Randall ViT-L inference requires a CUDA-capable PyTorch installation")
    net = DoRA_Sam(sam, args.rank).cuda()
    net.load_dora_parameters(args.dora_ckpt)
    net.eval()
    
    # Multiclass always uses multimask_output=True
    multimask_output = True

    # Build color palette and write id2label.json with color mapping
    color_palette = get_color_palette(args.num_classes)
    id2label_with_colors = build_id2label_with_colors(args.num_classes, id2label, color_palette)
    id2label_out_path = output_dir / "id2label.json"
    with open(id2label_out_path, "w") as f:
        json.dump(id2label_with_colors, f, indent=2)
    logging.info(f"Wrote colorized id2label mapping to {id2label_out_path}")
    
    # Find all image files
    input_dir = Path(args.input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    iterator = input_dir.rglob("*") if args.recursive else input_dir.iterdir()
    image_files = sorted(
        (f for f in iterator if f.is_file() and f.suffix.lower() in image_extensions),
        key=lambda path: path.as_posix().lower(),
    )
    
    if not image_files:
        raise ValueError(f"No image files found in {args.input_dir}")
    
    logging.info(f"Found {len(image_files)} images to process")

    pbar = tqdm(total=len(image_files), desc="Processing images")
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        inferencer = TileBatchInferencer(
            net=net,
            img_size=args.img_size,
            multimask_output=multimask_output,
            infer_batch_size=args.infer_batch_size,
            output_dir=output_dir,
            color_palette=color_palette,
            jsonl_f=jsonl_f,
            pbar=pbar,
            input_root=input_dir,
        )

        # Preprocess one image at a time and enqueue its tiles; run model on global tile batches.
        for item in iter_image_tile_lists(image_files, input_dir, args.img_size, args.tile_threshold):
            if "error" in item:
                inferencer.on_image_error(item)
                continue

            inferencer.enqueue_image(item)
            inferencer.drain_full_batches()

        inferencer.flush()

    pbar.close()
    logging.info(f"Wrote JSONL manifest to {jsonl_path}")

    logging.info(f"Inference complete! Output saved to {output_dir}")


if __name__ == '__main__':
    main()
