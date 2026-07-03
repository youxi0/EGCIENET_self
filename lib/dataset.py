# coding=utf-8

import os

import cv2
import numpy as np
from torch.utils.data import Dataset

try:
    from lib import transform
except ImportError:
    import transform


mean_rgb = np.array([[[0.551, 0.619, 0.532]]]) * 255
std_rgb = np.array([[[0.241, 0.236, 0.244]]]) * 255

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def _find_by_stem(directory, stem, exts=IMAGE_EXTS):
    for ext in exts:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _read_color(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError('Could not read image: {}'.format(path))
    return image.astype(np.float32)


class Data(Dataset):
    """ISAEB dataset loader.

    Train mode returns RGB, defect mask, SAM edge teacher, original size, and name.
    Test mode returns RGB, original size, and name. Inference does not require SAM.
    """

    def __init__(self, root, mode='train', image_size=352, require_edge=True):
        self.root = root
        self.mode = mode
        self.require_edge = require_edge
        self.samples = []

        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'BlackWhite')
        edge_dir = os.path.join(root, 'Edge')

        if not os.path.isdir(image_dir):
            raise FileNotFoundError('Missing JPEGImages directory: {}'.format(image_dir))

        image_names = [
            name for name in sorted(os.listdir(image_dir))
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS
        ]

        for image_name in image_names:
            stem = os.path.splitext(image_name)[0]
            rgb_path = os.path.join(image_dir, image_name)
            mask_path = _find_by_stem(mask_dir, stem) if os.path.isdir(mask_dir) else None
            edge_path = _find_by_stem(edge_dir, stem) if os.path.isdir(edge_dir) else None

            if mode == 'train':
                if mask_path is None:
                    raise FileNotFoundError('Missing mask for {} in {}'.format(image_name, mask_dir))
                if require_edge and edge_path is None:
                    raise FileNotFoundError('Missing edge teacher for {} in {}'.format(image_name, edge_dir))

            self.samples.append({
                'rgb': rgb_path,
                'mask': mask_path,
                'edge': edge_path,
                'name': image_name,
            })

        if mode == 'train':
            self.transform = transform.Compose(
                transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                transform.Resize(image_size, image_size),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
            )
        elif mode == 'test':
            self.transform = transform.Compose(
                transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                transform.Resize(image_size, image_size),
                transform.ToTensor(),
            )
        else:
            raise ValueError('Unsupported mode: {}'.format(mode))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = _read_color(sample['rgb'])
        h, w = rgb.shape[:2]

        if self.mode == 'train':
            mask = _read_color(sample['mask'])
            if sample['edge'] is not None:
                edge = _read_color(sample['edge'])
            else:
                edge = np.zeros_like(rgb)
            rgb, mask, edge = self.transform(rgb, mask, edge)
            return rgb, mask, edge, (h, w), sample['name']

        mask = np.zeros_like(rgb)
        edge = np.zeros_like(rgb)
        rgb, _, _ = self.transform(rgb, mask, edge)
        return rgb, (h, w), sample['name']

    def __len__(self):
        return len(self.samples)


class CombinedDataset(Dataset):
    """Combine two training roots with the same ISAEB folder layout."""

    def __init__(self, root1, root2, mode='train', image_size=352, require_edge=True):
        if mode != 'train':
            raise ValueError('CombinedDataset is intended for training only.')
        self.datasets = [
            Data(root1, mode=mode, image_size=image_size, require_edge=require_edge),
            Data(root2, mode=mode, image_size=image_size, require_edge=require_edge),
        ]
        self.lengths = [len(dataset) for dataset in self.datasets]

    def __getitem__(self, idx):
        if idx < self.lengths[0]:
            return self.datasets[0][idx]
        return self.datasets[1][idx - self.lengths[0]]

    def __len__(self):
        return sum(self.lengths)
