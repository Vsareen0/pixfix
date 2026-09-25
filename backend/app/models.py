"""Database models (Section viii – Database Design, db_pixfix).

tbl_users  1 ── N  tbl_images
"""
from datetime import datetime, timezone
import enum

from sqlalchemy import func

from .extensions import bcrypt, db


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ImageStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class User(db.Model):
    __tablename__ = "tbl_users"

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)  # bcrypt hash, never plaintext
    is_admin = db.Column(db.Boolean, nullable=False, default=False, server_default="false")
    is_active = db.Column(db.Boolean, nullable=False, default=True, server_default="true")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, server_default=func.now())
    last_login_at = db.Column(db.DateTime, nullable=True)

    images = db.relationship(
        "ImageLog", back_populates="user", cascade="all, delete-orphan", lazy="dynamic"
    )

    def set_password(self, raw: str) -> None:
        self.password = bcrypt.generate_password_hash(raw).decode("utf-8")

    def check_password(self, raw: str) -> bool:
        return bcrypt.check_password_hash(self.password, raw)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "is_active": self.is_active,
            "created_at": _iso(self.created_at),
            "last_login_at": _iso(self.last_login_at),
        }


class ImageLog(db.Model):
    __tablename__ = "tbl_images"

    img_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("tbl_users.user_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    original_filename = db.Column(db.String(255), nullable=True)
    original_path = db.Column(db.String(255), nullable=False)
    mask_path = db.Column(db.String(255), nullable=True)
    result_path = db.Column(db.String(255), nullable=True)
    status = db.Column(
        db.Enum(ImageStatus, name="image_status",
                values_callable=lambda e: [m.value for m in e]),
        nullable=False, default=ImageStatus.PENDING, index=True,
    )
    model_used = db.Column(db.String(20), nullable=True)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    file_size = db.Column(db.BigInteger, nullable=True)  # bytes, all files for this job
    processing_time = db.Column(db.Float, nullable=True)  # seconds
    error_message = db.Column(db.Text, nullable=True)
    upload_time = db.Column(db.DateTime, nullable=False, default=utcnow,
                            server_default=func.now(), index=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", back_populates="images")

    def to_dict(self) -> dict:
        return {
            "img_id": self.img_id,
            "user_id": self.user_id,
            "original_filename": self.original_filename,
            "status": self.status.value if self.status else None,
            "model_used": self.model_used,
            "width": self.width,
            "height": self.height,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "upload_time": _iso(self.upload_time),
            "completed_at": _iso(self.completed_at),
            "has_result": bool(self.result_path),
            "original_url": f"/api/images/{self.img_id}/file/original",
            "mask_url": f"/api/images/{self.img_id}/file/mask" if self.mask_path else None,
            "result_url": f"/api/images/{self.img_id}/file/result" if self.result_path else None,
        }


def _iso(dt):
    return dt.isoformat() + "Z" if dt else None
