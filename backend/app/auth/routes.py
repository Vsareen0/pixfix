"""Module 1: Authentication & User Management.

Input validation -> Hash Password -> Check DB -> Generate Token -> Grant Access.
"""
import re

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, current_user, jwt_required

from ..extensions import db
from ..models import User, utcnow

bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\- ]{3,50}$")


def _token_for(user: User) -> str:
    return create_access_token(identity=str(user.user_id),
                               additional_claims={"is_admin": user.is_admin})


@bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    errors = {}
    if not USERNAME_RE.match(username):
        errors["username"] = "3–50 characters: letters, numbers, space, _ . -"
    if not EMAIL_RE.match(email) or len(email) > 100:
        errors["email"] = "Enter a valid email address"
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        errors["password"] = "At least 8 characters, including a letter and a number"
    if errors:
        return jsonify(error="Validation failed", fields=errors), 400

    if User.query.filter_by(email=email).first():
        return jsonify(error="An account with this email already exists",
                       fields={"email": "Already registered"}), 409

    user = User(username=username, email=email)
    user.set_password(password)
    user.last_login_at = utcnow()
    db.session.add(user)
    db.session.commit()
    return jsonify(token=_token_for(user), user=user.to_dict()), 201


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify(error="Email and password are required"), 400

    user = User.query.filter_by(email=email).first()
    # Same message for unknown email / wrong password to avoid account enumeration.
    if not user or not user.check_password(password):
        return jsonify(error="Invalid email or password"), 401
    if not user.is_active:
        return jsonify(error="This account has been deactivated"), 403

    user.last_login_at = utcnow()
    db.session.commit()
    return jsonify(token=_token_for(user), user=user.to_dict())


@bp.get("/me")
@jwt_required()
def me():
    return jsonify(user=current_user.to_dict())
