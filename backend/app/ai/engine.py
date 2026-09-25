"""AI_Engine (Class Diagram, Fig. 7): loadModel / preprocess / predict / postprocess.

Three interchangeable backends:
  * lama   – pretrained LaMa (best quality, ~200 MB weights)
  * gan    – PixFix's own gated-conv GAN (trained with app/ai/gan/train.py)
  * patchmatch – classical exemplar-based fill: copies real patches (no weights needed)
  * opencv – classical Telea fast-marching diffusion (no weights needed, fastest)

"auto" picks the best backend that is available. Models are loaded lazily, once,
and shared between worker threads (inference is guarded by a lock).
"""
import logging
import threading

import cv2
import numpy as np
from flask import Flask
from PIL import Image

from .postprocessing import blend_into
from .preprocessing import context_crop, dilate_mask

log = logging.getLogger(__name__)

_engine_lock = threading.Lock()


def get_engine(app: Flask) -> "AIEngine":
    """One engine per Flask app, created on first use."""
    with _engine_lock:
        if "pixfix_engine" not in app.extensions:
            app.extensions["pixfix_engine"] = AIEngine(app.config)
        return app.extensions["pixfix_engine"]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class OpenCVInpainter:
    name = "opencv"

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        radius = max(3, int(np.sqrt((mask > 0).sum()) * 0.02))
        return cv2.inpaint(img, mask, min(radius, 15), cv2.INPAINT_TELEA)


class AIEngine:
    PRIORITY = ("lama", "gan", "patchmatch", "opencv")

    def __init__(self, config):
        self.cfg = config
        self._models = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.device = self._pick_device(config.get("AI_DEVICE", "auto"))
        configured = config.get("AI_BACKEND", "auto")
        avail = self.available_backends()
        self.default_backend = configured if configured in avail else avail[0]
        log.info("AI engine ready. device=%s available=%s default=%s",
                 self.device, avail, self.default_backend)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _pick_device(name):
        if not _torch_available():
            return "cpu"
        import torch
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def available_backends(self) -> list:
        out = []
        if _torch_available():
            if self.cfg["LAMA_MODEL_PATH"].exists():
                out.append("lama")
            if self.cfg["GAN_MODEL_PATH"].exists():
                out.append("gan")
        out += ["patchmatch", "opencv"]
        return out

    def load_model(self, backend: str):
        with self._load_lock:
            if backend in self._models:
                return self._models[backend]
            log.info("Loading %s model...", backend)
            if backend == "lama":
                from .lama import LamaInpainter
                model = LamaInpainter(self.cfg["LAMA_MODEL_PATH"], self.device,
                                      self.cfg["LAMA_MAX_SIDE"])
            elif backend == "gan":
                from .gan.inference import GanInpainter
                model = GanInpainter(self.cfg["GAN_MODEL_PATH"], self.device,
                                     self.cfg["GAN_INPUT_SIZE"])
            elif backend == "patchmatch":
                from .patchmatch import PatchMatchInpainter
                model = PatchMatchInpainter(max_side=self.cfg.get("PATCHMATCH_MAX_SIDE", 800))
            elif backend == "opencv":
                model = OpenCVInpainter()
            else:
                raise ValueError(f"Unknown backend {backend}")
            self._models[backend] = model
            return model

    def resolve_backend(self, requested) -> str:
        if not requested or requested == "auto":
            return self.default_backend
        if requested not in self.available_backends():
            raise RuntimeError(f"Model '{requested}' is not installed on this server "
                               f"(available: {', '.join(self.available_backends())})")
        return requested

    # --------------------------------------------------------------- pipeline
    def preprocess(self, image: Image.Image, mask: np.ndarray, backend: str):
        img = np.array(image.convert("RGB"))
        mask = (mask > 127).astype(np.uint8) * 255
        mask = dilate_mask(mask, px=max(2, round(min(img.shape[:2]) / 300)))
        # LaMa has a huge receptive field, so give it lots of context; the
        # 256px GAN does better with a tighter crop at higher effective resolution.
        context = {"lama": 2.0, "gan": 1.0, "patchmatch": 1.0, "opencv": 0.5}[backend]
        min_side = {"lama": 512, "gan": 128, "patchmatch": 160, "opencv": 64}[backend]
        crop = context_crop(mask, context=context, min_side=min_side)
        return img, mask, crop

    def predict(self, model, img_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
        with self._infer_lock:
            return model(np.ascontiguousarray(img_crop), np.ascontiguousarray(mask_crop))

    def postprocess(self, img, generated_crop, mask, crop) -> Image.Image:
        feather_px = max(2, round(min(img.shape[:2]) / 400))
        return Image.fromarray(blend_into(img, generated_crop, mask, crop, feather_px))

    def inpaint(self, image: Image.Image, mask: np.ndarray, backend=None):
        """Full pipeline. Returns (result PIL image, backend actually used)."""
        backend = self.resolve_backend(backend)
        model = self.load_model(backend)
        img, mask, crop = self.preprocess(image, mask, backend)
        img_c = img[crop.y0:crop.y1, crop.x0:crop.x1]
        mask_c = mask[crop.y0:crop.y1, crop.x0:crop.x1]
        generated = self.predict(model, img_c, mask_c)
        return self.postprocess(img, generated, mask, crop), backend
