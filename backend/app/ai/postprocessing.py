"""Module 5: Post-processing & Output.

Denormalize -> Resize back to original resolution -> Blend the generated region
into the original high-res image, so pixels outside the mask are untouched.
"""
import cv2
import numpy as np

from .preprocessing import Crop


def to_uint8(tensor, value_range: str) -> np.ndarray:
    """1x3xHxW tensor -> HxWx3 uint8."""
    t = tensor.detach().float().cpu()[0]
    if value_range == "-1..1":
        t = (t + 1) / 2
    arr = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255 + 0.5).astype(np.uint8)


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft alpha in [0,1]: 1 inside the mask, smooth falloff just outside it."""
    m = (mask > 127).astype(np.float32)
    if radius <= 0:
        return m
    k = 2 * radius + 1
    grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    soft = cv2.GaussianBlur(grown, (k, k), 0)
    return np.maximum(soft, m)


def blend_into(original: np.ndarray, generated_crop: np.ndarray, mask: np.ndarray,
               crop: Crop, feather_px: int) -> np.ndarray:
    """Resize `generated_crop` to the crop size and alpha-blend it into `original`."""
    out = original.copy()
    gh, gw = generated_crop.shape[:2]
    if (gw, gh) != (crop.w, crop.h):
        interp = cv2.INTER_CUBIC if crop.w > gw else cv2.INTER_AREA
        generated_crop = cv2.resize(generated_crop, (crop.w, crop.h), interpolation=interp)

    region = out[crop.y0:crop.y1, crop.x0:crop.x1].astype(np.float32)
    alpha = feather(mask[crop.y0:crop.y1, crop.x0:crop.x1], feather_px)[..., None]
    blended = region * (1 - alpha) + generated_crop.astype(np.float32) * alpha
    out[crop.y0:crop.y1, crop.x0:crop.x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out
