"""CARPK vehicle counting. Box annotations are converted to centre points.

Expected layout of ``root``: ``Images/``, ``Annotations/``, ``ImageSets/{train,test}.txt``.
"""
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

SIZE_WH = (640, 384)
TRAIN_CROP = 224


class CARPK(Dataset):
    def __init__(self, root, split="train", prompt="car"):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split
        self.prompt = prompt
        list_file = os.path.join(root, "ImageSets", f"{split}.txt")
        with open(list_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.points = {}
        for im_id in self.ids:
            with open(os.path.join(root, "Annotations", f"{im_id}.txt")) as f:
                boxes = [[float(v) for v in line.split()[:4]] for line in f if line.strip()]
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            self.points[im_id] = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        im_id = self.ids[idx]
        image = Image.open(os.path.join(self.root, "Images", f"{im_id}.png")).convert("RGB")
        W, H = image.size
        points = self.points[im_id] * np.array([SIZE_WH[0] / W, SIZE_WH[1] / H], dtype=np.float32)
        image = TF.to_tensor(image.resize(SIZE_WH, Image.BILINEAR))
        if self.split == "train":
            image, points = self._augment(image, points)
        return image, torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2), self.prompt, im_id

    @staticmethod
    def _augment(image, points):
        W, H = SIZE_WH
        points = points.copy()
        if random.random() < 0.5:
            image = (image + torch.randn_like(image) * 0.1).clamp(0, 1)
            if random.random() > 0.5:
                image = TF.hflip(image)
                points[:, 0] = W - points[:, 0]
            if random.random() > 0.5:
                image = TF.vflip(image)
                points[:, 1] = H - points[:, 1]
        x0 = random.randint(0, W - TRAIN_CROP)
        y0 = random.randint(0, H - TRAIN_CROP)
        image = TF.crop(image, y0, x0, TRAIN_CROP, TRAIN_CROP)
        keep = (points[:, 0] >= x0) & (points[:, 0] <= x0 + TRAIN_CROP) & \
               (points[:, 1] >= y0) & (points[:, 1] <= y0 + TRAIN_CROP)
        return image, points[keep] - np.array([x0, y0], dtype=np.float32)
