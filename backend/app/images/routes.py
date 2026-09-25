"""Image upload / processing / history / download endpoints (ImageController)."""
from flask import Blueprint, abort, current_app, jsonify, request, send_file
from flask_jwt_extended import current_user, jwt_required
from PIL import Image

from .. import tasks
from ..extensions import db
from ..models import ImageLog, ImageStatus
from ..services import storage
from ..services.validation import ValidationError, allowed_extension, load_image, load_mask

bp = Blueprint("images", __name__)

VALID_BACKENDS = {"auto", "lama", "gan", "patchmatch", "opencv"}


def _own_job_or_404(img_id: int) -> ImageLog:
    job = db.session.get(ImageLog, img_id)
    if job is None or (job.user_id != current_user.user_id and not current_user.is_admin):
        abort(404, description="Image not found")
    return job


@bp.post("/process")
@jwt_required()
def process():
    """POST /api/images/process  (multipart)

    Fields:
      image      – JPG/PNG file  (or `source_id` to re-edit a previous result)
      mask       – PNG, white/opaque = area to remove
      model      – optional: auto | lama | gan | patchmatch | opencv
      source_id  – optional: id of a previous job; its result (or original) is used as input
    """
    cfg = current_app.config
    backend = (request.form.get("model") or "auto").lower()
    if backend not in VALID_BACKENDS:
        return jsonify(error=f"Unknown model '{backend}'"), 400

    try:
        source_id = request.form.get("source_id")
        if source_id:
            src = _own_job_or_404(int(source_id))
            src_rel = src.result_path or src.original_path
            image = Image.open(storage.abs_path(src_rel)).convert("RGB")
            original_filename = src.original_filename
        else:
            f = request.files.get("image")
            if f is None or not f.filename:
                return jsonify(error="No image uploaded"), 400
            if not allowed_extension(f.filename, cfg["ALLOWED_EXTENSIONS"]):
                return jsonify(error="Only .jpg, .jpeg and .png files are allowed"), 400
            image = load_image(f.read(), cfg["MAX_IMAGE_PIXELS"])
            original_filename = f.filename[:255]

        m = request.files.get("mask")
        mask = load_mask(m.read() if m else b"", image.size, cfg["MAX_IMAGE_PIXELS"])
    except ValidationError as e:
        return jsonify(error=str(e)), 400
    except ValueError:
        return jsonify(error="Invalid source_id"), 400

    job = ImageLog(
        user_id=current_user.user_id,
        original_filename=original_filename,
        original_path=storage.save_original(image),
        mask_path=storage.save_mask(mask),
        status=ImageStatus.PENDING,
        width=image.width,
        height=image.height,
    )
    db.session.add(job)
    db.session.commit()

    img_id = job.img_id
    tasks.submit(current_app._get_current_object(), img_id, backend)
    job = db.session.get(ImageLog, img_id)
    return jsonify(image=job.to_dict()), 202


@bp.get("")
@jwt_required()
def history():
    """Paginated history of the current user's jobs (View History use case)."""
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(max(request.args.get("per_page", 12, type=int), 1), 100)
    status = request.args.get("status")

    q = ImageLog.query.filter_by(user_id=current_user.user_id)
    if status in {s.value for s in ImageStatus}:
        q = q.filter(ImageLog.status == ImageStatus(status))
    pag = q.order_by(ImageLog.upload_time.desc()).paginate(page=page, per_page=per_page,
                                                           error_out=False)
    return jsonify(items=[j.to_dict() for j in pag.items], page=pag.page,
                   pages=pag.pages, total=pag.total)


@bp.get("/<int:img_id>")
@jwt_required()
def get_one(img_id):
    return jsonify(image=_own_job_or_404(img_id).to_dict())


@bp.get("/<int:img_id>/file/<kind>")
@jwt_required()
def get_file(img_id, kind):
    job = _own_job_or_404(img_id)
    rel = {"original": job.original_path, "mask": job.mask_path,
           "result": job.result_path}.get(kind)
    if not rel:
        abort(404, description="File not available")
    path = storage.abs_path(rel)
    if not path.exists():
        abort(404, description="File missing on server")

    download = request.args.get("download") == "1"
    base = (job.original_filename or f"image_{job.img_id}").rsplit(".", 1)[0]
    name = f"{base}_pixfix.png" if kind == "result" else f"{base}_{kind}.png"
    return send_file(path, mimetype="image/png", as_attachment=download, download_name=name,
                     max_age=0)


@bp.delete("/<int:img_id>")
@jwt_required()
def delete(img_id):
    job = _own_job_or_404(img_id)
    if job.status == ImageStatus.PROCESSING:
        return jsonify(error="Cannot delete while processing"), 409
    storage.delete(job.original_path, job.mask_path, job.result_path)
    db.session.delete(job)
    db.session.commit()
    return jsonify(ok=True)
