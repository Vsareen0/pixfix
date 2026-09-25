"""Input validation (Security Mechanism 1).

Checks the extension *and* the actual decoded content, so a script renamed to
.png is rejected. Pillow's decompression-bomb guard is also applied.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}


class ValidationError(ValueError):
    pass


def allowed_extension(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def load_image(data: bytes, max_pixels: int) -> Image.Image:
    """Decode an uploaded image, verify it, return an RGB PIL image with EXIF rotation applied."""
    if not data:
        raise ValidationError("Empty file")
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        probe = Image.open(BytesIO(data))
        fmt = probe.format
        probe.verify()  # structural integrity check
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise ValidationError("File is not a valid JPG or PNG image") from e
    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(f"Unsupported image format: {fmt}. Only JPG and PNG are allowed")

    img = Image.open(BytesIO(data))
    if img.width * img.height > max_pixels:
        raise ValidationError("Image resolution is too large")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        # Composite transparency onto white so the model sees a sensible background.
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def load_mask(data: bytes, size: tuple, max_pixels: int) -> np.ndarray:
    """Decode a mask PNG -> uint8 binary array (255 = remove, 0 = keep), resized to `size` (w, h).

    Accepts either a white-on-black mask or a transparent PNG where painted pixels are opaque.
    """
    if not data:
        raise ValidationError("Mask is required")
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        m = Image.open(BytesIO(data))
        m.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValidationError("Mask is not a valid image") from e

    if m.mode in ("RGBA", "LA"):
        arr = np.array(m.getchannel("A"))
    else:
        arr = np.array(m.convert("L"))
    mask_img = Image.fromarray(arr)
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.NEAREST)
    mask = (np.array(mask_img) > 127).astype(np.uint8) * 255
    if mask.sum() == 0:
        raise ValidationError("Mask is empty – paint over the area you want to remove")
    return mask
