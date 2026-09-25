"""File-system storage for originals, masks and results.

Paths stored in the DB are *relative* to STORAGE_DIR, and filenames are random
UUIDs – user-supplied names never touch the file system (prevents path traversal).
"""
import uuid
from pathlib import Path

import numpy as np
from flask import current_app
from PIL import Image


def _root() -> Path:
    return Path(current_app.config["STORAGE_DIR"])


def abs_path(rel: str) -> Path:
    root = _root().resolve()
    p = (root / rel).resolve()
    if root not in p.parents:
        raise ValueError("Invalid storage path")
    return p


def _new_rel(kind: str, ext: str) -> str:
    return f"{kind}/{uuid.uuid4().hex}.{ext}"


def save_original(img: Image.Image) -> str:
    rel = _new_rel("originals", "png")
    img.save(abs_path(rel), format="PNG", optimize=False)
    return rel


def save_mask(mask: np.ndarray) -> str:
    rel = _new_rel("masks", "png")
    Image.fromarray(mask).save(abs_path(rel), format="PNG")
    return rel


def save_result(img: Image.Image) -> str:
    rel = _new_rel("results", "png")
    img.save(abs_path(rel), format="PNG")
    return rel


def size_of(*rels) -> int:
    total = 0
    for rel in rels:
        if rel:
            p = abs_path(rel)
            if p.exists():
                total += p.stat().st_size
    return total


def delete(*rels) -> None:
    for rel in rels:
        if rel:
            try:
                abs_path(rel).unlink(missing_ok=True)
            except ValueError:
                pass
