"""Module 3: Pre-processing.

Resize -> Normalize (0-255 -> 0-1 or -1..1) -> Tensor.

To keep quality on high-resolution photos we do not squash the whole image into
the model's input size. Instead we crop a context window around the masked
region, run the model on that crop, and paste the result back (Module 5).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow the mask a few pixels so object edges/halos are also replaced."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def context_crop(mask: np.ndarray, context: float = 1.5, min_side: int = 256) -> Crop:
    """Bounding box of the mask, expanded to include surrounding context."""
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    side = int(max(bw, bh) * (1 + context))
    side = max(side, min_side)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    # Window of the wanted size centred on the mask, clamped to stay inside the image.
    sw, sh = min(side, w), min(side, h)
    nx0 = int(np.clip(cx - sw // 2, 0, w - sw))
    ny0 = int(np.clip(cy - sh // 2, 0, h - sh))
    nx1, ny1 = nx0 + sw, ny0 + sh
    # Never cut through the mask itself (can happen when the mask is wider than `side` allows).
    nx0, ny0 = min(nx0, int(x0)), min(ny0, int(y0))
    nx1, ny1 = max(nx1, int(x1)), max(ny1, int(y1))
    return Crop(int(nx0), int(ny0), int(nx1), int(ny1))


def resize_max_side(img: np.ndarray, max_side: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)


def pad_to_multiple(img: np.ndarray, mult: int):
    """Reflect-pad H, W up to a multiple of `mult`. Returns (padded, (h, w))."""
    h, w = img.shape[:2]
    ph, pw = (-h) % mult, (-w) % mult
    if ph == 0 and pw == 0:
        return img, (h, w)
    pad = ((0, ph), (0, pw)) + (((0, 0),) if img.ndim == 3 else ())
    return np.pad(img, pad, mode="reflect" if min(h, w) > 1 else "edge"), (h, w)


def letterbox_square(img: np.ndarray, size: int, interp):
    """Aspect-preserving resize to fit `size`, then reflect-pad to size x size.

    Returns (square, (new_h, new_w)) so the padding can be removed later.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad = ((0, size - nh), (0, size - nw)) + (((0, 0),) if img.ndim == 3 else ())
    mode = "reflect" if min(nh, nw) > 1 and size - nh < nh and size - nw < nw else "edge"
    return np.pad(resized, pad, mode=mode), (nh, nw)


def to_tensor(img: np.ndarray, mask: np.ndarray, value_range: str, device):
    """uint8 HxWx3 image + uint8 HxW mask -> (1x3xHxW float, 1x1xHxW float {0,1})."""
    import torch

    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if value_range == "-1..1":
        x = x * 2 - 1
    m = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return x.to(device), m.to(device)
