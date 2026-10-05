"""FSC-147 with FSC-147-D text descriptions (e.g. "the apples") as prompts.

Expected layout of ``root``::

    images_384_VarV2/
    annotation_FSC147_384.json
    Train_Test_Val_FSC147.json
    ImageClasses_FSC147.txt
    FSC-147-D.json            # text descriptions, from the CounTX repository
"""
import json
import os
import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from imgaug.augmentables import Keypoint, KeypointsOnImage
from PIL import Image
from torch.utils.data import Dataset

CROP = 384
to_tensor = T.ToTensor()
color_blur = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.15, saturation=0.2, hue=0.2),
    T.GaussianBlur(kernel_size=(7, 9)),
])


def resize_to_multiple_of_32(image, points):
    """Resizes a PIL image down to the nearest multiple of 32 and rescales the points."""
    W, H = image.size
    new_w, new_h = 32 * (W // 32), 32 * (H // 32)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    points = points * np.array([new_w / W, new_h / H], dtype=np.float32)
    return image, points


class FSC147(Dataset):
    """
    Args:
        root: dataset root.
        split: 'train', 'val' or 'test'.
        prompt: 'description' (FSC-147-D, default) or 'class' (category name).
    """

    def __init__(self, root, split="train", prompt="description"):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split
        self.im_dir = os.path.join(root, "images_384_VarV2")
        with open(os.path.join(root, "annotation_FSC147_384.json")) as f:
            self.annotations = json.load(f)
        with open(os.path.join(root, "Train_Test_Val_FSC147.json")) as f:
            self.ids = json.load(f)[split]
        if prompt == "description":
            with open(os.path.join(root, "FSC-147-D.json")) as f:
                self.prompts = {k: v["text_description"] for k, v in json.load(f).items()}
        elif prompt == "class":
            with open(os.path.join(root, "ImageClasses_FSC147.txt")) as f:
                self.prompts = dict((line.split()[0], " ".join(line.split()[1:])) for line in f if line.strip())
        else:
            raise ValueError(f"Unknown prompt type {prompt!r}")
        self.augment = TrainAugmentation() if split == "train" else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        im_id = self.ids[idx]
        image = Image.open(os.path.join(self.im_dir, im_id)).convert("RGB")
        points = np.array(self.annotations[im_id]["points"], dtype=np.float32).reshape(-1, 2)
        image, points = resize_to_multiple_of_32(image, points)
        if self.augment is not None:
            image, points = self.augment(image, points)
        else:
            image = to_tensor(image)
        return image.float(), torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2), self.prompts[im_id], im_id


class TrainAugmentation:
    """Training augmentation used for RT-Counter on FSC-147.

    With probability 0.3 a 2x2 mosaic of four random crops of the same image is
    built; with probability 0.2 the image is perturbed (Gaussian noise, colour
    jitter + blur, random affine, horizontal/vertical flips); otherwise it is kept.
    Non-mosaic images are then randomly cropped to 384 x 384.
    """

    def __call__(self, image, points):
        W, H = image.size
        image = to_tensor(image)
        p = random.random()
        if p < 0.3:
            return self.mosaic(image, points, W, H)
        if p < 0.5:
            image, points = self.perturb(image, points, W, H)
        return self.random_crop(image, points, W)

    @staticmethod
    def perturb(image, points, W, H):
        image = (image + torch.randn_like(image) * 0.1).clamp(0, 1)
        image = color_blur(image)

        array = image.permute(1, 2, 0).numpy()
        kps = KeypointsOnImage([Keypoint(x=min(W - 1, int(x)), y=min(H - 1, int(y))) for x, y in points],
                               shape=array.shape)
        affine = iaa.Affine(rotate=(-15, 15), scale=(0.8, 1.2), shear=(-10, 10),
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode=ia.ALL)
        array, kps = affine(image=array, keypoints=kps)
        points = np.array([(kp.x, kp.y) for kp in kps if not kp.is_out_of_image(array)], dtype=np.float32)
        points = points.reshape(-1, 2)
        points = points[(points[:, 0] <= W) & (points[:, 1] <= H)]
        image = to_tensor(array)

        if random.random() > 0.5:
            image = TF.hflip(image)
            points[:, 0] = W - points[:, 0]
        if random.random() > 0.5:
            image = TF.vflip(image)
            points[:, 1] = H - points[:, 1]
        return image, points

    @staticmethod
    def random_crop(image, points, W):
        start = random.randint(0, max(0, W - CROP))
        image = TF.crop(image, 0, start, CROP, CROP)
        keep = (points[:, 0] >= start) & (points[:, 0] <= start + CROP) & (points[:, 1] >= 0) & (points[:, 1] <= CROP)
        points = points[keep].copy()
        points[:, 0] -= start
        return image, points

    @staticmethod
    def mosaic(image, points, W, H):
        half = CROP // 2
        blend = random.randint(10, 20)
        size = half + 2 * blend
        tiles, tile_points = [], []
        for _ in range(4):
            length = random.randint(150, 384)
            x0 = random.randint(0, W - length)
            y0 = random.randint(0, H - length)
            tile = T.Resize((size, size))(TF.crop(image, y0, x0, length, length))
            tiles.append(color_blur(tile))
            keep = (points[:, 0] >= x0) & (points[:, 0] <= x0 + length) & \
                   (points[:, 1] >= y0) & (points[:, 1] <= y0 + length)
            tile_points.append((points[keep] - np.array([x0, y0], dtype=np.float32)) * (size / length))

        # Tiles 0/1 form the left column (top/bottom) and tiles 2/3 the right column.
        for i, pts in enumerate(tile_points):
            keep = (pts[:, 0] >= blend) & (pts[:, 0] <= size - blend) & (pts[:, 1] >= blend) & (pts[:, 1] <= size - blend)
            pts = pts[keep] - blend
            pts[:, 1] += half * (i % 2)
            pts[:, 0] += half * (i // 2)
            tile_points[i] = pts

        def stack(a, b, dim):
            # Joins the inner parts of a and b along ``dim`` (1 = rows, 2 = columns) and
            # cross-fades the ``blend`` lines on each side of the seam.
            out = torch.cat((a.narrow(dim, blend, half), b.narrow(dim, blend, half)), dim)
            for i in range(blend):
                w_far, w_near = (blend - i) / (2 * blend), (i + blend) / (2 * blend)
                out.select(dim, half + i).copy_(
                    a.select(dim, size - 1 - blend + i) * w_far + out.select(dim, half + i) * w_near)
                out.select(dim, half - 1 - i).copy_(
                    b.select(dim, blend - i) * w_far + out.select(dim, half - 1 - i) * w_near)
            return out.clamp(0, 1)

        left = stack(tiles[0], tiles[1], dim=1)
        right = stack(tiles[2], tiles[3], dim=1)
        image = stack(left, right, dim=2)
        return image, np.concatenate(tile_points).reshape(-1, 2)
