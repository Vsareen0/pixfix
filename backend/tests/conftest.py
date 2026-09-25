import io

import numpy as np
import pytest
from PIL import Image

from app import create_app
from app.config import TestConfig
from app.extensions import db
from app.models import User


@pytest.fixture
def app(tmp_path):
    class Cfg(TestConfig):
        STORAGE_DIR = tmp_path / "storage"
        MODELS_DIR = tmp_path / "models"
        LAMA_MODEL_PATH = tmp_path / "models" / "none.pt"
        GAN_MODEL_PATH = tmp_path / "models" / "none.pth"

    app = create_app(Cfg)
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, email="alice@example.com", password="secret123", username="alice"):
    r = client.post("/api/auth/register", json={"email": email, "password": password,
                                                "username": username})
    return r


@pytest.fixture
def token(client):
    return register(client).get_json()["token"]


@pytest.fixture
def auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth(app, client):
    with app.app_context():
        u = User(email="admin@example.com", username="admin", is_admin=True)
        u.set_password("adminpass1")
        db.session.add(u)
        db.session.commit()
    t = client.post("/api/auth/login", json={"email": "admin@example.com",
                                             "password": "adminpass1"}).get_json()["token"]
    return {"Authorization": f"Bearer {t}"}


def png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    buf.seek(0)
    return buf


def sample_image(h=120, w=160):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 140, 90)
    img[40:80, 60:100] = (250, 20, 20)  # the "object" to remove
    return img


def sample_mask(h=120, w=160):
    m = np.zeros((h, w), np.uint8)
    m[36:84, 56:104] = 255
    return m
