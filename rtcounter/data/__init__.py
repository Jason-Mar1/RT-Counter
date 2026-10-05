import torch
import torch.nn.functional as F

from .carpk import CARPK
from .fsc147 import FSC147
from .rec8k import REC8K

DATASETS = {"fsc147": FSC147, "carpk": CARPK, "rec8k": REC8K}


def build_dataset(name, root, split):
    return DATASETS[name.lower()](root, split)


def collate_fn(batch):
    """Zero-pads images (bottom/right) to a common size that is a multiple of 32."""
    images, points, prompts, ids = zip(*batch)
    h = max(img.shape[1] for img in images)
    w = max(img.shape[2] for img in images)
    h, w = -(-h // 32) * 32, -(-w // 32) * 32
    images = torch.stack([F.pad(img, (0, w - img.shape[2], 0, h - img.shape[1])) for img in images])
    return images, list(points), list(prompts), list(ids)
