"""REC-8K referring expression counting.

Expected layout of ``root``: ``rec-8k/`` (images) and ``anno/{annotations,splits}.json``.
"""
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

CROP = 384


def preprocess_caption(caption):
    caption = caption.lower().strip()
    return caption if caption.endswith(".") else caption + "."


class REC8K(Dataset):
    def __init__(self, root, split="train", max_size=640):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split
        self.max_size = max_size
        with open(os.path.join(root, "anno", "annotations.json"), encoding="utf-8") as f:
            self.annotations = json.load(f)
        with open(os.path.join(root, "anno", "splits.json"), encoding="utf-8") as f:
            self.samples = json.load(f)[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, expression = self.samples[idx]
        image = Image.open(os.path.join(self.root, "rec-8k", image_name)).convert("RGB")
        anno = self.annotations.get(image_name, {}).get(expression, {})
        points = np.array(anno.get("points", []), dtype=np.float32).reshape(-1, 2)

        # Longest side <= max_size, both sides multiples of 32.
        W, H = image.size
        scale = min(self.max_size / max(W, H), 1.0)
        new_w, new_h = max(32, int(W * scale) // 32 * 32), max(32, int(H * scale) // 32 * 32)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        points = points * np.array([new_w / W, new_h / H], dtype=np.float32)
        if self.split == "train":
            image, points = self._random_crop(image, points)
        return TF.to_tensor(image), torch.as_tensor(points).reshape(-1, 2), preprocess_caption(expression), image_name

    @staticmethod
    def _random_crop(image, points):
        W, H = image.size
        if W < CROP or H < CROP:
            scale = CROP / min(W, H)
            W, H = int(W * scale), int(H * scale)
            image = image.resize((W, H), Image.LANCZOS)
            points = points * scale
        left = random.randint(0, W - CROP)
        top = random.randint(0, H - CROP)
        image = TF.crop(image, top, left, CROP, CROP)
        keep = (points[:, 0] >= left) & (points[:, 0] < left + CROP) & \
               (points[:, 1] >= top) & (points[:, 1] < top + CROP)
        return image, points[keep] - np.array([left, top], dtype=np.float32)
