"""Administrator module: Manage Users & Logs, and the three reports in the proposal:
  1. User Activity Report
  2. Processing Time Analysis Report
  3. Failure/Error Logs
"""
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, abort, jsonify, request
from flask_jwt_extended import current_user, jwt_required
from sqlalchemy import func, or_

from ..extensions import db
from ..models import ImageLog, ImageStatus, User, utcnow
from ..services import storage

bp = Blueprint("admin", __name__)


def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403, description="Administrator access required")
        return fn(*args, **kwargs)
    return wrapper


def _start_of_today():
    now = utcnow()
    return datetime(now.year, now.month, now.day)


@bp.get("/stats")
@admin_required
def stats():
    """User Activity Report (summary cards)."""
    today = _start_of_today()
    active_ids = {uid for (uid,) in db.session.query(ImageLog.user_id)
                  .filter(ImageLog.upload_time >= today).distinct()}
    active_ids |= {uid for (uid,) in db.session.query(User.user_id)
                   .filter(User.last_login_at >= today)}

    by_status = dict(db.session.query(ImageLog.status, func.count()).group_by(ImageLog.status).all())
    storage_bytes = db.session.query(func.coalesce(func.sum(ImageLog.file_size), 0)).scalar()
    avg_time = db.session.query(func.avg(ImageLog.processing_time)) \
        .filter(ImageLog.status == ImageStatus.DONE).scalar()

    return jsonify(
        total_users=User.query.count(),
        active_today=len(active_ids),
        total_images=ImageLog.query.count(),
        images_today=ImageLog.query.filter(ImageLog.upload_time >= today).count(),
        status_counts={s.value: by_status.get(s, 0) for s in ImageStatus},
        total_storage_bytes=int(storage_bytes or 0),
        avg_processing_time=round(avg_time, 3) if avg_time else None,
    )


@bp.get("/users")
@admin_required
def list_users():
    search = (request.args.get("q") or "").strip()
    counts = dict(db.session.query(ImageLog.user_id, func.count()).group_by(ImageLog.user_id).all())
    sizes = dict(db.session.query(ImageLog.user_id, func.coalesce(func.sum(ImageLog.file_size), 0))
                 .group_by(ImageLog.user_id).all())
    q = User.query
    if search:
        like = f"%{search}%"
        q = q.filter(or_(User.email.ilike(like), User.username.ilike(like)))
    users = []
    for u in q.order_by(User.created_at.desc()).all():
        d = u.to_dict()
        d["image_count"] = counts.get(u.user_id, 0)
        d["storage_bytes"] = int(sizes.get(u.user_id, 0) or 0)
        users.append(d)
    return jsonify(items=users)


@bp.patch("/users/<int:user_id>")
@admin_required
def update_user(user_id):
    user = db.session.get(User, user_id) or abort(404, description="User not found")
    data = request.get_json(silent=True) or {}
    if user.user_id == current_user.user_id and (data.get("is_active") is False
                                                 or data.get("is_admin") is False):
        return jsonify(error="You cannot deactivate or demote yourself"), 400
    if "is_active" in data:
        user.is_active = bool(data["is_active"])
    if "is_admin" in data:
        user.is_admin = bool(data["is_admin"])
    db.session.commit()
    return jsonify(user=user.to_dict())


@bp.delete("/users/<int:user_id>")
@admin_required
def delete_user(user_id):
    user = db.session.get(User, user_id) or abort(404, description="User not found")
    if user.user_id == current_user.user_id:
        return jsonify(error="You cannot delete your own account"), 400
    for job in user.images:
        storage.delete(job.original_path, job.mask_path, job.result_path)
    db.session.delete(user)
    db.session.commit()
    return jsonify(ok=True)


@bp.get("/logs")
@admin_required
def logs():
    """Processing Log / Processing Time Analysis report (+ error logs with ?status=failed)."""
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(max(request.args.get("per_page", 25, type=int), 1), 200)
    status = request.args.get("status")
    days = request.args.get("days", type=int)

    q = db.session.query(ImageLog, User.username, User.email).join(User)
    if status in {s.value for s in ImageStatus}:
        q = q.filter(ImageLog.status == ImageStatus(status))
    if days:
        q = q.filter(ImageLog.upload_time >= utcnow() - timedelta(days=days))
    q = q.order_by(ImageLog.upload_time.desc())

    total = q.count()
    rows = q.offset((page - 1) * per_page).limit(per_page).all()
    items = []
    for job, username, email in rows:
        d = job.to_dict()
        d["username"], d["email"] = username, email
        items.append(d)
    return jsonify(items=items, total=total, page=page,
                   pages=max((total + per_page - 1) // per_page, 1))


@bp.get("/reports/processing-time")
@admin_required
def processing_time_report():
    """Aggregated processing time per model and per day (last N days)."""
    days = request.args.get("days", 14, type=int)
    since = utcnow() - timedelta(days=days)
    done = ImageLog.query.filter(ImageLog.status == ImageStatus.DONE, ImageLog.upload_time >= since)

    per_model = [
        {"model": m or "unknown", "count": c, "avg": round(a or 0, 3),
         "min": round(mn or 0, 3), "max": round(mx or 0, 3)}
        for m, c, a, mn, mx in db.session.query(
            ImageLog.model_used, func.count(), func.avg(ImageLog.processing_time),
            func.min(ImageLog.processing_time), func.max(ImageLog.processing_time))
        .filter(ImageLog.status == ImageStatus.DONE, ImageLog.upload_time >= since)
        .group_by(ImageLog.model_used).all()
    ]

    # Per-day buckets computed in Python so it works on both SQLite and PostgreSQL.
    buckets = {}
    for job in done.with_entities(ImageLog.upload_time, ImageLog.processing_time):
        key = job.upload_time.date().isoformat()
        b = buckets.setdefault(key, [0, 0.0])
        b[0] += 1
        b[1] += job.processing_time or 0
    per_day = [{"date": k, "count": v[0], "avg": round(v[1] / v[0], 3)}
               for k, v in sorted(buckets.items())]

    failed = ImageLog.query.filter(ImageLog.status == ImageStatus.FAILED,
                                   ImageLog.upload_time >= since).count()
    return jsonify(days=days, per_model=per_model, per_day=per_day, failed=failed)
