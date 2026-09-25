import io

import numpy as np
from PIL import Image

from conftest import png_bytes, register, sample_image, sample_mask


def _process(client, auth, img=None, mask=None, name="photo.png", **extra):
    data = {"image": (png_bytes(img if img is not None else sample_image()), name),
            "mask": (png_bytes(mask if mask is not None else sample_mask()), "mask.png")}
    data.update(extra)
    return client.post("/api/images/process", data=data, headers=auth,
                       content_type="multipart/form-data")


def test_process_removes_object(client, auth):
    r = _process(client, auth)
    assert r.status_code == 202, r.get_json()
    job = r.get_json()["image"]
    assert job["status"] == "done" and job["model_used"] == "opencv"
    assert job["processing_time"] is not None

    r = client.get(job["result_url"], headers=auth)
    assert r.status_code == 200 and r.mimetype == "image/png"
    out = np.array(Image.open(io.BytesIO(r.data)).convert("RGB")).astype(int)
    # The red square should now be (close to) the green background.
    centre = out[55:65, 75:85].mean(axis=(0, 1))
    assert centre[0] < 120 and centre[1] > 100
    # Pixels far outside the mask are untouched.
    assert (out[5:20, 5:20] == [30, 140, 90]).all()


def test_download_header(client, auth):
    job = _process(client, auth).get_json()["image"]
    r = client.get(job["result_url"] + "?download=1", headers=auth)
    assert "attachment" in r.headers["Content-Disposition"]
    assert "photo_pixfix.png" in r.headers["Content-Disposition"]


def test_rejects_bad_extension_and_fake_image(client, auth):
    r = _process(client, auth, name="evil.exe")
    assert r.status_code == 400
    data = {"image": (io.BytesIO(b"<?php system($_GET['c']); ?>"), "evil.png"),
            "mask": (png_bytes(sample_mask()), "m.png")}
    r = client.post("/api/images/process", data=data, headers=auth,
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "valid" in r.get_json()["error"]


def test_rejects_gif_even_with_png_extension(client, auth):
    buf = io.BytesIO()
    Image.fromarray(sample_image()).save(buf, format="GIF")
    buf.seek(0)
    data = {"image": (buf, "x.png"), "mask": (png_bytes(sample_mask()), "m.png")}
    r = client.post("/api/images/process", data=data, headers=auth,
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_empty_mask_rejected(client, auth):
    r = _process(client, auth, mask=np.zeros((120, 160), np.uint8))
    assert r.status_code == 400 and "empty" in r.get_json()["error"].lower()


def test_mask_is_resized_and_rgba_supported(client, auth):
    rgba = np.zeros((60, 80, 4), np.uint8)
    rgba[18:42, 28:52] = (255, 0, 0, 255)
    r = _process(client, auth, mask=rgba)
    assert r.status_code == 202 and r.get_json()["image"]["status"] == "done"


def test_unavailable_model(client, auth):  # lama weights are absent in tests
    r = _process(client, auth, model="lama")
    job = r.get_json()["image"]
    assert job["status"] == "failed" and "not installed" in job["error_message"]


def test_history_and_isolation(client, auth):
    _process(client, auth)
    _process(client, auth)
    r = client.get("/api/images", headers=auth).get_json()
    assert r["total"] == 2

    other = register(client, email="bob@example.com", username="bob").get_json()["token"]
    bob = {"Authorization": f"Bearer {other}"}
    assert client.get("/api/images", headers=bob).get_json()["total"] == 0
    img_id = r["items"][0]["img_id"]
    assert client.get(f"/api/images/{img_id}", headers=bob).status_code == 404
    assert client.get(f"/api/images/{img_id}/file/original", headers=bob).status_code == 404


def test_chain_edit_from_previous_result(client, auth):
    first = _process(client, auth).get_json()["image"]
    data = {"source_id": str(first["img_id"]),
            "mask": (png_bytes(sample_mask()), "m.png")}
    r = client.post("/api/images/process", data=data, headers=auth,
                    content_type="multipart/form-data")
    assert r.status_code == 202 and r.get_json()["image"]["status"] == "done"


def test_delete(client, auth):
    job = _process(client, auth).get_json()["image"]
    assert client.delete(f"/api/images/{job['img_id']}", headers=auth).status_code == 200
    assert client.get(f"/api/images/{job['img_id']}", headers=auth).status_code == 404
