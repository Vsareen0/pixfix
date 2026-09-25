"""Pretrained LaMa (Large Mask Inpainting, Suvorov et al., WACV 2022).

Loads the TorchScript export of `big-lama` (the same file used by lama-cleaner /
IOPaint). Contract: image 1x3xHxW float in [0,1], mask 1x1xHxW float {0,1},
H and W divisible by 8; output 1x3xHxW in [0,1].

Download with:  python scripts/download_models.py
"""
import cv2
import numpy as np

from .postprocessing import to_uint8
from .preprocessing import pad_to_multiple, resize_max_side, to_tensor


class LamaInpainter:
    name = "lama"

    def __init__(self, model_path, device, max_side: int = 1024):
        import warnings

        import torch

        self.device = device
        self.max_side = max_side
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.jit deprecation notice
            self.model = torch.jit.load(str(model_path), map_location=device).eval()

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        h0, w0 = img.shape[:2]
        img_s = resize_max_side(img, self.max_side, cv2.INTER_AREA)
        mask_s = resize_max_side(mask, self.max_side, cv2.INTER_NEAREST)
        img_p, (h, w) = pad_to_multiple(img_s, 8)
        mask_p, _ = pad_to_multiple(mask_s, 8)

        x, m = to_tensor(img_p, mask_p, "0..1", self.device)
        with torch.inference_mode():
            y = self.model(x, m)
        out = to_uint8(y, "0..1")[:h, :w]
        if (w, h) != (w0, h0):
            out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_CUBIC)
        return out
