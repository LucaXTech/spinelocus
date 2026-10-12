"""On-the-fly crops avoid storing thousands of duplicated PNG files."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import read_label
from .geometry import crop, to_local

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_tensor(image):
    array = np.array(image.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(((array - MEAN) / STD).transpose(2, 0, 1).copy())


class LandmarkCrops(Dataset):
    def __init__(self, manifest, root, split, cfg):
        self.root, self.cfg, self.split, self.epoch = Path(root), cfg, split, 0
        self.entries = []
        for row in manifest[manifest.split == split].itertuples():
            xy, vis = read_label(self.root / row.label, row.width, row.height)
            for i in range(len(xy)):
                if vis[i] > 0:
                    self.entries.append((row.image, xy[i]))
        if not self.entries:
            raise ValueError(f"No visible landmarks in {split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, gt = self.entries[index]
        # Epoch/index seed keeps worker scheduling from changing augmentation.
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.get("seed", 42), self.epoch, index]))
        c = int(self.cfg["crop_size"])
        center, size = np.array(gt, dtype=float), c
        if self.split == "train":
            fraction = self.cfg.get("translate_fraction", 0.2)
            center += rng.uniform(-fraction * c, fraction * c, 2)
            size = round(c * rng.uniform(*self.cfg.get("scale_range", [0.9, 1.2])))
        elif self.cfg.get("validation_jitter", 0):
            # Fixed validation perturbation; separately identify this protocol in config.
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.get("seed", 42), 0, index]))
            center += rng.uniform(-self.cfg["validation_jitter"] * c, self.cfg["validation_jitter"] * c, 2)
        with Image.open(self.root / name) as im:
            patch, bounds = crop(im.convert("RGB"), center, size, c)
        local = to_local(gt, bounds, c, self.cfg.get("coordinate_convention", "pixel_centers"))
        if np.any(local < 0) or np.any(local >= c):
            raise ValueError("GT fell outside crop. Reduce jitter; targets are never silently clamped.")
        if self.split == "train" and rng.random() < self.cfg.get("horizontal_flip", 0.5):
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            local[0] = c - 1 - local[0]
        return image_tensor(patch), torch.tensor(local, dtype=torch.float32)
