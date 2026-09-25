"""Training data: any folder of photos + randomly generated free-form masks.

Suggested datasets (Section: Dataset Collection):
  * Places365 (scenes)  http://places2.csail.mit.edu/
  * CelebA-HQ (faces)
  * Paris StreetView
  * Or simply your own photo folder for a small demo model.
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def random_free_form_mask(h, w, max_strokes=6, max_vertex=12, max_width=None, rng=random):
    """DeepFill-v2-style random brush strokes (what a user draws with a brush)."""
    mask = np.zeros((h, w), np.uint8)
    max_width = max_width or max(8, int(min(h, w) * 0.1))
    for _ in range(rng.randint(1, max_strokes)):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        width = rng.randint(max(4, max_width // 3), max_width)
        for _ in range(rng.randint(1, max_vertex)):
            angle = rng.uniform(0, 2 * math.pi)
            length = rng.randint(10, max(11, min(h, w) // 4))
            nx = int(np.clip(x + length * math.cos(angle), 0, w - 1))
            ny = int(np.clip(y + length * math.sin(angle), 0, h - 1))
            cv2.line(mask, (x, y), (nx, ny), 255, width)
            cv2.circle(mask, (nx, ny), width // 2, 255, -1)
            x, y = nx, ny
    return mask


def random_box_mask(h, w, rng=random):
    """Rectangular hole (simulates removing a compact object)."""
    mask = np.zeros((h, w), np.uint8)
    bh, bw = rng.randint(h // 8, h // 2), rng.randint(w // 8, w // 2)
    y, x = rng.randint(0, h - bh), rng.randint(0, w - bw)
    mask[y:y + bh, x:x + bw] = 255
    return mask


def random_mask(h, w, rng=random):
    r = rng.random()
    if r < 0.6:
        return random_free_form_mask(h, w, rng=rng)
    if r < 0.85:
        return random_box_mask(h, w, rng=rng)
    return np.maximum(random_free_form_mask(h, w, rng=rng), random_box_mask(h, w, rng=rng))


class InpaintingDataset(Dataset):
    def __init__(self, root, size=256, train=True, fixed_masks=False):
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")
        self.size = size
        self.train = train
        self.fixed_masks = fixed_masks  # deterministic masks for validation

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = self.size
        # Resize shorter side, then random (train) / centre (val) crop.
        scale = s / min(w, h) * (random.uniform(1.0, 1.25) if self.train else 1.0)
        img = img.resize((max(s, round(w * scale)), max(s, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        if self.train:
            x, y = random.randint(0, w - s), random.randint(0, h - s)
        else:
            x, y = (w - s) // 2, (h - s) // 2
        img = img.crop((x, y, x + s, y + s))
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return np.array(img)

    def __getitem__(self, idx):
        img = self._load(self.files[idx])
        rng = random.Random(idx) if self.fixed_masks else random
        mask = random_mask(self.size, self.size, rng=rng)
        x = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
        m = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return x, m
