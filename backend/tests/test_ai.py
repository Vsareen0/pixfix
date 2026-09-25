import numpy as np
import pytest

from app.ai.postprocessing import blend_into, feather
from app.ai.preprocessing import context_crop, letterbox_square, pad_to_multiple

torch = pytest.importorskip("torch")


def test_context_crop_contains_mask_and_stays_in_bounds():
    m = np.zeros((500, 800), np.uint8)
    m[480:500, 780:800] = 255  # bottom-right corner
    c = context_crop(m, context=1.0, min_side=128)
    assert 0 <= c.x0 < c.x1 <= 800 and 0 <= c.y0 < c.y1 <= 500
    assert c.x0 <= 780 and c.x1 >= 800 and c.y0 <= 480 and c.y1 >= 500
    assert c.w >= 128 and c.h >= 128


def test_pad_and_letterbox():
    img = np.zeros((101, 203, 3), np.uint8)
    p, (h, w) = pad_to_multiple(img, 8)
    assert p.shape[0] % 8 == 0 and p.shape[1] % 8 == 0 and (h, w) == (101, 203)
    sq, (nh, nw) = letterbox_square(img, 256, 1)
    assert sq.shape[:2] == (256, 256) and nw == 256 and nh == 127


def test_blend_keeps_unmasked_pixels():
    from app.ai.preprocessing import Crop
    orig = np.full((64, 64, 3), 10, np.uint8)
    gen = np.full((32, 32, 3), 200, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[24:40, 24:40] = 255
    out = blend_into(orig, gen, mask, Crop(16, 16, 48, 48), feather_px=2)
    assert (out[30, 30] == 200).all()
    assert (out[0:10, 0:10] == 10).all()
    assert feather(mask, 2).max() == 1.0


def test_generator_and_discriminator_shapes():
    from app.ai.gan.networks import InpaintGenerator, PatchDiscriminator
    G, D = InpaintGenerator(base=16), PatchDiscriminator(base=16)
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    m = torch.zeros(2, 1, 128, 128)
    m[:, :, 32:96, 32:96] = 1
    raw, comp = G(x, m)
    assert raw.shape == x.shape and comp.abs().max() <= 1
    # known pixels are copied through untouched
    assert torch.allclose(comp[:, :, :32], x[:, :, :32])
    assert D(comp, m).shape == (2, 1, 4, 4)


def test_train_smoke_and_load_in_engine(tmp_path):
    from PIL import Image
    from app.ai.gan import train
    from app.ai.gan.inference import GanInpainter

    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)).save(data / f"{i}.png")
    out = tmp_path / "ck"
    train.main(["--data", str(data), "--out", str(out), "--size", "64", "--base", "8",
                "--epochs", "1", "--batch-size", "2", "--workers", "0", "--device", "cpu"])
    assert (out / "pixfix_gan.pth").exists() and (out / "history.csv").exists()

    g = GanInpainter(out / "pixfix_gan.pth", torch.device("cpu"))
    img = rng.integers(0, 255, (70, 100, 3), dtype=np.uint8)
    mask = np.zeros((70, 100), np.uint8)
    mask[20:40, 30:60] = 255
    res = g(img, mask)
    assert res.shape == img.shape and res.dtype == np.uint8


def test_context_crop_centred_on_interior_mask():
    """Regression: an odd-sized window used to be shifted to the image edge, missing the mask."""
    m = np.zeros((512, 512), np.uint8)
    m[343:428, 129:214] = 255
    c = context_crop(m, context=0.5, min_side=64)
    assert c.x0 <= 129 and c.x1 >= 214 and c.y0 <= 343 and c.y1 >= 428
    assert c.x0 > 50 and c.x1 < 300  # stays around the mask, not snapped to a border


def test_patchmatch_copies_texture_into_hole():
    from app.ai.patchmatch import PatchMatchInpainter

    # vertical stripes, period 8; a square "object" covers part of them
    img = np.zeros((96, 128, 3), np.uint8)
    img[:, (np.arange(128) % 8) < 4] = (220, 60, 40)
    img[:, (np.arange(128) % 8) >= 4] = (30, 90, 200)
    truth = img.copy()
    img[30:66, 50:86] = (0, 255, 0)
    mask = np.zeros((96, 128), np.uint8)
    mask[30:66, 50:86] = 255

    out = PatchMatchInpainter()(img, mask)
    assert out.shape == img.shape
    assert (out[mask == 0] == img[mask == 0]).all()          # known pixels untouched
    assert out[mask > 0, 1].max() < 200                        # green object gone
    err = np.abs(out.astype(int) - truth.astype(int))[mask > 0].mean()
    assert err < 25, err                                       # stripes reconstructed


def test_patchmatch_backend_via_api_engine(app):
    from PIL import Image
    from app.ai.engine import get_engine

    engine = get_engine(app)
    assert "patchmatch" in engine.available_backends()
    img = Image.fromarray(np.full((80, 100, 3), 120, np.uint8))
    mask = np.zeros((80, 100), np.uint8)
    mask[30:50, 40:60] = 255
    out, used = engine.inpaint(img, mask, backend="patchmatch")
    assert used == "patchmatch" and out.size == (100, 80)
