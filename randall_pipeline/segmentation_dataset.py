import os
import random
import cv2
# Suppress OpenCV TIFF warnings - 0 = SILENT, 1 = FATAL, 2 = ERROR, 3 = WARN, 4 = INFO
try:
    cv2.setLogLevel(0)  # SILENT - suppress all warnings
except AttributeError:
    # Fallback for older OpenCV versions
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import h5py
import numpy as np
import torch
import warnings
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat


CACHE_VERSION = 2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(0, 1))

    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis-1).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res, do_random=True):
        self.output_size = output_size
        self.low_res = low_res
        self.do_random = do_random

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.do_random:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        c, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # nearest neighbor for labels
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)  # nearest neighbor
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))  # Keep as int64 for class IDs
        low_res_label = torch.from_numpy(low_res_label.astype(np.int64))  # Keep as int64 for class IDs
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class hm_dataset_multiclass(Dataset):
    def __init__(self, base_dir, split, transform=None, id2label=None,
                 cache_dir=None, cache_img_size=None, cache_low_res=None,
                 cache_read=True, cache_write=True):
        """
        Multiclass dataset loader for class ID masks.
        
        Args:
            base_dir: Base directory containing train/test splits and annotation subdirectory
            split: Split name (e.g., 'train', 'test')
            transform: Optional transform to apply
            id2label: Optional dict mapping class IDs to label names (for logging)
        """
        self.transform = transform
        self.split = split
        self.img_dir = os.path.join(base_dir, split)
        self.label_dir = os.path.join(base_dir, 'annotation', split)
        self.file_names = os.listdir(os.path.join(base_dir, split))
        self.id2label = id2label
        self.base_dir = os.path.abspath(base_dir)
        self.cache_dir = None
        self.cache_img_size = cache_img_size
        self.cache_low_res = cache_low_res
        self.cache_read = cache_read
        self.cache_write = cache_write
        if cache_dir is not None:
            split_cache = os.path.join(cache_dir, split)
            os.makedirs(split_cache, exist_ok=True)
            self.cache_dir = split_cache
        
        print(f"Initialized multiclass dataset with {len(self.file_names)} samples")
        if self.id2label:
            print(f"Class labels: {self.id2label}")

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def _cache_path(self, stem):
        if self.cache_dir is None or self.cache_img_size is None or self.cache_low_res is None:
            return None
        return os.path.join(
            self.cache_dir,
            f"{stem}_img{self.cache_img_size}_low{self.cache_low_res}.npz"
        )

    def _load_cache(self, cache_path, img_path, label_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
        except Exception as exc:
            warnings.warn(f"Ignoring unreadable cache file {cache_path}: {exc}")
            return None

        required = {
            'image',
            'label',
            'low_res_label',
            'cache_version',
            'base_dir',
            'split',
            'img_path',
            'label_path',
        }
        if not required.issubset(set(cached.files)):
            return None

        cache_version = int(np.asarray(cached['cache_version']).item())
        base_dir = str(np.asarray(cached['base_dir']).item())
        split = str(np.asarray(cached['split']).item())
        cached_img_path = str(np.asarray(cached['img_path']).item())
        cached_label_path = str(np.asarray(cached['label_path']).item())

        if (
            cache_version != CACHE_VERSION
            or base_dir != self.base_dir
            or split != self.split
            or cached_img_path != os.path.abspath(img_path)
            or cached_label_path != os.path.abspath(label_path)
        ):
            return None

        image = cached['image'].astype(np.float32)
        label = cached['label'].astype(np.int64)
        low_res_label = cached['low_res_label'].astype(np.int64)
        return image, label, low_res_label

    def _resize_and_lowres(self, image, label):
        # image: CHW float in [0,1], label: HW int
        c, h, w = image.shape
        if self.cache_img_size is None or self.cache_low_res is None:
            return image, label, None
        if h != self.cache_img_size or w != self.cache_img_size:
            image_hwc = np.transpose(image, (1, 2, 0))
            image_hwc = cv2.resize(image_hwc, (self.cache_img_size, self.cache_img_size), interpolation=cv2.INTER_LINEAR)
            image = np.transpose(image_hwc, (2, 0, 1))
            label = cv2.resize(label, (self.cache_img_size, self.cache_img_size), interpolation=cv2.INTER_NEAREST)
        low_res_label = cv2.resize(label, (self.cache_low_res, self.cache_low_res), interpolation=cv2.INTER_NEAREST)
        return image, label, low_res_label

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        stem = os.path.splitext(file_name)[0]
        cache_path = self._cache_path(stem)
        img_path = os.path.join(self.img_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        if cache_path and self.cache_read and os.path.exists(cache_path):
            cached_sample = self._load_cache(cache_path, img_path, label_path)
            if cached_sample is not None:
                image, label, low_res_label = cached_sample
                sample = {
                    'image': torch.from_numpy(image),
                    'label': torch.from_numpy(label),
                    'low_res_label': torch.from_numpy(low_res_label),
                    'case_name': stem,
                }
                return sample

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            raise ValueError(f"Failed to load label: {label_path}")

        # Handle image: convert to RGB and normalize
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.transpose(image, (2, 0, 1)) / 255.0  # c,h,w, normalize to [0,1]

        # Handle label: masks should be single-channel with class IDs
        if len(label.shape) == 3:
            raise ValueError(f"Expected single-channel class ID mask, but got {len(label.shape)}-channel mask. "
                           f"Masks must be class IDs, not RGB.")
        elif len(label.shape) != 2:
            raise ValueError(f"Unexpected label shape: {label.shape}. Expected 2D array (H, W) with class IDs.")

        label = label.astype(np.int64)

        low_res_label = None
        if self.cache_dir is not None and self.cache_img_size is not None and self.cache_low_res is not None:
            image, label, low_res_label = self._resize_and_lowres(image, label)
            if self.cache_write and cache_path is not None:
                tmp_path = cache_path + f".tmp{os.getpid()}.npz"
                np.savez(
                    tmp_path,
                    image=image.astype(np.float16),
                    label=label.astype(np.int64),
                    low_res_label=low_res_label.astype(np.int64),
                    cache_version=np.array(CACHE_VERSION, dtype=np.int64),
                    base_dir=np.array(self.base_dir),
                    split=np.array(self.split),
                    img_path=np.array(os.path.abspath(img_path)),
                    label_path=np.array(os.path.abspath(label_path)),
                    label_classes=np.unique(label).astype(np.int64),
                )
                os.replace(tmp_path, cache_path)

        sample = {'image': image, 'label': label}
        if low_res_label is not None:
            sample['low_res_label'] = low_res_label
        if self.transform:
            sample = self.transform(sample)
        else:
            # Convert to tensors if no transform is provided
            sample['image'] = torch.from_numpy(sample['image'].astype(np.float32))
            sample['label'] = torch.from_numpy(sample['label'].astype(np.int64))
            if 'low_res_label' in sample:
                sample['low_res_label'] = torch.from_numpy(sample['low_res_label'].astype(np.int64))
        sample['case_name'] = stem
        return sample
