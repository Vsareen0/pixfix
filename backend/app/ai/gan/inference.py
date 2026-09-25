"""Run the trained PixFix GAN generator on a crop."""
import cv2
import numpy as np

from ..postprocessing import to_uint8
from ..preprocessing import letterbox_square, to_tensor


class GanInpainter:
    name = "gan"

    def __init__(self, model_path, device, input_size: int = 256):
        import torch
        from .networks import InpaintGenerator

        ck = torch.load(str(model_path), map_location=device, weights_only=True)
        cfg = ck.get("config", {})
        self.input_size = int(cfg.get("input_size", input_size))
        self.model = InpaintGenerator(base=int(cfg.get("base", 32)))
        self.model.load_state_dict(ck["generator"] if "generator" in ck else ck)
        self.model.to(device).eval()
        self.device = device

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        h0, w0 = img.shape[:2]
        s = self.input_size
        img_sq, (nh, nw) = letterbox_square(img, s, cv2.INTER_AREA)
        mask_sq, _ = letterbox_square(mask, s, cv2.INTER_NEAREST)
        # padding area is "known" context; the generator must not treat it as a hole
        mask_sq[nh:, :] = 0
        mask_sq[:, nw:] = 0

        x, m = to_tensor(img_sq, mask_sq, "-1..1", self.device)
        with torch.inference_mode():
            _, comp = self.model(x, m)
        out = to_uint8(comp, "-1..1")[:nh, :nw]
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_CUBIC)
