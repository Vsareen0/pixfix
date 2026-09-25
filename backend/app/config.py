"""Application configuration.

All settings can be overridden through environment variables (or a `.env`
file in the backend directory). Defaults are chosen so the app runs locally
with zero setup: SQLite database, files under ./storage, OpenCV fallback if no
model weights are present.
"""
import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=int(os.getenv("JWT_EXPIRES_HOURS", "12")))

    # Database: PostgreSQL in production (docker-compose), SQLite for local dev.
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'pixfix_dev.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # File storage (Data Tier: file system for images)
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(40_000_000)))  # ~40 MP
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    # AI engine
    # "auto" -> LaMa if weights exist, else custom GAN if weights exist, else PatchMatch
    AI_BACKEND = os.getenv("AI_BACKEND", "auto")
    MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
    LAMA_MODEL_PATH = Path(os.getenv("LAMA_MODEL_PATH", MODELS_DIR / "big-lama.pt"))
    GAN_MODEL_PATH = Path(os.getenv("GAN_MODEL_PATH", MODELS_DIR / "pixfix_gan.pth"))
    AI_DEVICE = os.getenv("AI_DEVICE", "auto")  # auto | cpu | cuda
    LAMA_MAX_SIDE = int(os.getenv("LAMA_MAX_SIDE", "1024"))
    GAN_INPUT_SIZE = int(os.getenv("GAN_INPUT_SIZE", "256"))
    # PatchMatch works on the context crop, downscaled to at most this many pixels per side
    PATCHMATCH_MAX_SIDE = int(os.getenv("PATCHMATCH_MAX_SIDE", "800"))

    # Async processing
    PROCESS_ASYNC = _bool("PROCESS_ASYNC", True)
    WORKER_THREADS = int(os.getenv("WORKER_THREADS", "2"))

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {}
    PROCESS_ASYNC = False
    AI_BACKEND = "opencv"
    JWT_SECRET_KEY = "test-secret-key-that-is-long-enough-for-hs256"
