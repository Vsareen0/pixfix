"""Background job runner.

High-resolution images can take seconds to process, so jobs are queued on a
small thread pool and the client polls the job status (Scope: "batches will be
processed asynchronously"). In tests (PROCESS_ASYNC=False) jobs run inline.
"""
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask
from PIL import Image

from .extensions import db
from .models import ImageLog, ImageStatus, utcnow
from .services import storage

log = logging.getLogger(__name__)
_executor = None


def _get_executor(app: Flask) -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=app.config["WORKER_THREADS"],
                                       thread_name_prefix="pixfix-worker")
    return _executor


def submit(app: Flask, img_id: int, backend: str | None) -> None:
    if app.config["PROCESS_ASYNC"]:
        _get_executor(app).submit(_run_with_context, app, img_id, backend)
    else:
        _run(app, img_id, backend)


def _run_with_context(app: Flask, img_id: int, backend: str | None) -> None:
    with app.app_context():
        _run(app, img_id, backend)


def _run(app: Flask, img_id: int, backend: str | None) -> None:
    from .ai.engine import get_engine

    job = db.session.get(ImageLog, img_id)
    if job is None:
        return
    job.status = ImageStatus.PROCESSING
    db.session.commit()

    start = time.perf_counter()
    try:
        image = Image.open(storage.abs_path(job.original_path)).convert("RGB")
        mask = np.array(Image.open(storage.abs_path(job.mask_path)).convert("L"))
        engine = get_engine(app)
        result, used = engine.inpaint(image, mask, backend=backend)

        job.result_path = storage.save_result(result)
        job.model_used = used
        job.status = ImageStatus.DONE
        job.error_message = None
    except Exception as e:  # noqa: BLE001 – every failure must be logged on the job
        log.error("Job %s failed: %s\n%s", img_id, e, traceback.format_exc())
        job.status = ImageStatus.FAILED
        job.error_message = f"{type(e).__name__}: {e}"[:2000]
    finally:
        job.processing_time = round(time.perf_counter() - start, 3)
        job.completed_at = utcnow()
        job.file_size = storage.size_of(job.original_path, job.mask_path, job.result_path)
        db.session.commit()
        db.session.remove()
