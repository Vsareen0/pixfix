# PixFix – AI-Driven Image Restoration and Object Removal Tool

PixFix is a full-stack web app for removing unwanted objects, photobombers and scratches from photos. You upload a JPG/PNG, paint over the area you want gone, and a deep-learning inpainting model fills it in with plausible content.

| | |
|---|---|
| **Presentation tier** | React 18 + Vite, HTML5 Canvas mask editor |
| **Application tier** | Python 3.10+, Flask REST API, PyTorch, OpenCV |
| **Data tier** | PostgreSQL (SQLite for local dev), file-system image storage |
| **AI models** | PixFix GAN (our own gated-conv GAN, trainable) · pretrained LaMa · PatchMatch (classical, exemplar-based) · OpenCV Telea |

---

## Features

- **Authentication:** register/login with JWT; bcrypt password hashing; accounts can be deactivated.
- **Upload:** JPG/PNG only. The server checks the extension and also decodes the file, so a renamed script is rejected. Size and resolution limits apply.
- **Interactive masking:** brush and eraser, adjustable size, undo/redo, clear, keyboard shortcuts.
- **AI inference:** choose a model or let *Auto* pick the best one installed. Jobs run asynchronously, and the UI polls until they finish.
- **High-resolution aware:** the model runs on a context crop around the mask, and the result is blended back into the untouched full-resolution original.
- **Result view:** before/after slider, download as PNG, *Redraw mask* (keeps your strokes), *Keep editing this result* (chained edits).
- **History:** every job with thumbnail, status, model and time taken. You can re-download, edit again or delete.
- **Admin:** user management (enable/disable, promote, delete) plus the three reports from the proposal:
  - User Activity (total users, active today, storage used)
  - Processing Time Analysis (per model, per day)
  - Failure/Error Log (CSV export)

---

## Quick start (local development)

Prerequisites: Python 3.10+, Node 18+. A GPU is optional; everything runs on CPU, just slower.

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
#   NVIDIA GPU? install the CUDA build of torch first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

python scripts/download_models.py      # pretrained LaMa (~196 MB) -> backend/models/big-lama.pt
cp .env.example .env                   # optional: edit settings
flask --app wsgi create-admin          # prompts for email / username / password
python wsgi.py                         # API on http://localhost:5000
```

With no `DATABASE_URL` set, the backend uses SQLite (`backend/pixfix_dev.db`). To use PostgreSQL:

```bash
createdb db_pixfix
export DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost:5432/db_pixfix
```

The tables are created automatically on startup; `backend/schema.sql` holds the equivalent DDL for your report.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                            # http://localhost:5173  (proxies /api -> :5000)
```

Open http://localhost:5173, sign up, and upload an image.

### 3. Docker (all three tiers)

```bash
# put big-lama.pt (and/or pixfix_gan.pth) in backend/models/ first
docker compose up --build              # http://localhost:8080
docker compose exec backend flask --app wsgi create-admin
```

---

## The AI core

```
backend/app/ai/
├── engine.py          AIEngine: load_model / preprocess / predict / postprocess, backend selection
├── preprocessing.py   mask dilation, context crop, resize/pad, normalise, to-tensor   (Module 3)
├── postprocessing.py  de-normalise, resize back, feathered blend into original       (Module 5)
├── lama.py            pretrained LaMa (TorchScript)
├── patchmatch.py      classical exemplar-based fill: multi-scale PatchMatch + EM voting
└── gan/               PixFix GAN                                                    (Module 4)
    ├── networks.py    Generator (gated-conv encoder–decoder, dilated bottleneck) + SN-PatchGAN discriminator
    ├── losses.py      L1 hole/valid reconstruction, hinge adversarial, optional VGG perceptual
    ├── dataset.py     image folder + random free-form / box masks
    ├── train.py       training loop, checkpoints, sample grids, PSNR validation, history.csv
    └── inference.py   runs the trained generator inside the app
```

**Pipeline (DFD Level 2):**

1. Dilate the mask a few pixels.
2. Crop a context window around it.
3. Resize or pad the crop to the model input size and normalise it.
4. Run the generator.
5. De-normalise, resize back to the crop size, feather-blend into the original, and save a PNG.

Pixels outside the mask are never changed.

### Backends

| Backend | What it is | When to use |
|---|---|---|
| `lama` | Pretrained *Large Mask Inpainting* (Suvorov et al., 2022) | Best quality out of the box. Recommended for demos. |
| `gan` | **PixFix GAN**, our own model trained with `train.py` | Shows the GAN architecture described in the proposal. Quality depends on how long you train it. |
| `patchmatch` | Exemplar-based completion: Wexler et al. 2007 with PatchMatch search (Barnes et al. 2009), in vectorised NumPy | No weights needed. Copies real texture, so it is much better than Telea on fabric, grass, sky and wood. It is weaker where structure must be continued (edges, object outlines). About 2–7 s per edit on CPU. |
| `opencv` | Telea fast-marching diffusion (classical, no learning) | Instant. Fine for thin scratches, but smears large holes. Useful as the simplest baseline. |

`AI_BACKEND=auto` (the default) picks the best installed backend in the order LaMa → GAN → PatchMatch → OpenCV.

### How PatchMatch works

Diffusion methods (Telea) push boundary colours inward, so they cannot create texture. PatchMatch instead fills the hole with **real 7×7 patches copied from the known part of the image**, chosen so that overlapping patches agree:

1. **Pyramid.** Start at a coarse scale (≥ 64 px) where the hole is only a few pixels thick.
2. **Onion-peel initial fill** at the coarsest scale. Fill ring by ring from the boundary inward, highest-confidence pixels first (Criminisi et al. 2004). Each pixel takes the best-matching source patch, compared on known pixels only, so the removed object never influences the result.
3. **EM iterations** at each scale:
   - *NNF step:* PatchMatch finds, for every patch touching the hole, the most similar fully-known patch (random search, plus propagation of good matches to neighbours).
   - *Vote step:* each hole pixel becomes the similarity-weighted average of what its overlapping patches suggest.
4. **Up-sample** the matches to the next scale and repeat, finishing at full working resolution (`PATCHMATCH_MAX_SIDE`, default 800 px on the context crop).

![Original · OpenCV Telea · PatchMatch · LaMa](docs/inpainting_comparison.png)

*Columns: original, OpenCV Telea, PatchMatch, LaMa (CPU times shown). PatchMatch beats Telea on texture (fabric, sky). LaMa is best where structure must be continued (the saucer rim).*

An optional Poisson (gradient-domain) blend is included (`PatchMatchInpainter(poisson=True)`). It is off by default: in our tests it also pulled leftover object pixels on the mask edge (glows, shadows) into the fill.

### Training the PixFix GAN

```bash
cd backend
# any folder of photos works; Places365 or Paris StreetView subsets are good choices for scenes
python -m app.ai.gan.train --data ./data/train --val-data ./data/val \
       --epochs 30 --batch-size 8 --perceptual
cp checkpoints/pixfix_gan_best.pth models/pixfix_gan.pth   # the app picks it up on restart
```

Each run writes the following to `checkpoints/`:

- `last.pth` and `best.pth` (resumable with `--resume`)
- `pixfix_gan.pth`, the generator-only file the app loads
- `samples/epoch_XXX.png` grids (masked input, output, ground truth), which are good figures for your report
- `history.csv` with losses and validation PSNR per epoch

On a CPU this is only practical as a smoke test. For usable results, train on a CUDA GPU with thousands of images for many hours. Until then, an under-trained GAN produces flat, blurry fills, which is expected.

---

## REST API

All endpoints except register, login and health need `Authorization: Bearer <token>`.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/register` | `{username, email, password}` → `{token, user}` |
| POST | `/api/auth/login` | `{email, password}` → `{token, user}` |
| GET | `/api/auth/me` | current user |
| POST | `/api/images/process` | multipart: `image` (or `source_id`), `mask`, `model` → job (202) |
| GET | `/api/images?page=&status=` | the user's history |
| GET | `/api/images/<id>` | job status (poll this) |
| GET | `/api/images/<id>/file/{original,mask,result}[?download=1]` | image files |
| DELETE | `/api/images/<id>` | delete a job and its files |
| GET | `/api/admin/stats` | User Activity report |
| GET | `/api/admin/users?q=` · PATCH/DELETE `/api/admin/users/<id>` | manage users |
| GET | `/api/admin/logs?status=&days=&page=` | processing log / error log |
| GET | `/api/admin/reports/processing-time?days=` | Processing Time Analysis |
| GET | `/api/health` | installed AI backends, device |

---

## Tests

```bash
cd backend && pytest          # 27 tests: auth, validation, processing, isolation, admin, AI pipeline, PatchMatch, GAN train+load
```

---

## How the code maps to the proposal

| Proposal section | Implementation |
|---|---|
| Module 1 – Authentication & User Management | `backend/app/auth/routes.py`, `models.User` |
| Module 2 – Image Input & Interaction | `frontend/src/components/MaskEditor.jsx`, `pages/Editor.jsx` |
| Module 3 – Pre-processing | `backend/app/ai/preprocessing.py` |
| Module 4 – AI Core (GAN) | `backend/app/ai/gan/` (+ `lama.py`, `patchmatch.py`) |
| Module 5 – Post-processing & Output | `backend/app/ai/postprocessing.py` |
| Class diagram: ImageController / AI_Engine / Database | `images/routes.py` · `ai/engine.py` · `models.py` + SQLAlchemy |
| DB design (`db_pixfix`, `tbl_users`, `tbl_images`) | `backend/app/models.py`, `backend/schema.sql` |
| Sequence diagram (`POST /api/process`) | `POST /api/images/process` → `tasks.py` → engine → DB |
| Reports (activity, processing time, errors) | `backend/app/admin/routes.py`, `frontend/src/pages/Admin.jsx` |
| Three-tier architecture | `docker-compose.yml` (nginx+React / Flask+PyTorch / PostgreSQL+volume) |
| Security mechanisms | See below |

**Security mechanisms**

- **Content validation:** uploads are checked for type by decoding the file, not only by extension (`services/validation.py`).
- **Password hashing:** bcrypt.
- **SQL injection:** all queries go through the SQLAlchemy ORM.
- **Safe storage paths:** files are saved under random UUID names, and paths are checked to stay inside the storage folder.
- **Access control:** users only see their own images (other users' images return 404); admin endpoints are role-checked.
- **Response headers:** nginx adds security headers.
- **HTTPS:** terminate TLS at nginx (for a local demo, use a self-signed or `mkcert` certificate).

**Additions to the proposal's schema**

- `status` has two extra values, `processing` and `failed`, so the async jobs and the error report can work.
- `tbl_images` has extra columns needed for the reports and the ERD's `mask_path`: `mask_path`, `model_used`, `width`, `height`, `file_size`, `processing_time`, `error_message`, `completed_at`.
- `tbl_users` has extra columns for admin management and the "active today" figure: `is_admin`, `is_active`, `last_login_at`.

---

## Project structure

```
pixfix/
├── backend/
│   ├── app/
│   │   ├── __init__.py        app factory, error handlers, `create-admin` CLI
│   │   ├── config.py          env-driven settings
│   │   ├── models.py          tbl_users, tbl_images
│   │   ├── tasks.py           async job runner (thread pool)
│   │   ├── auth/ images/ admin/   blueprints
│   │   ├── services/          validation, storage
│   │   └── ai/                engine, pre/post-processing, LaMa, PixFix GAN
│   ├── scripts/download_models.py
│   ├── tests/
│   ├── schema.sql  requirements.txt  wsgi.py  Dockerfile
├── frontend/
│   ├── src/ (pages, components, api.js, auth.jsx, styles.css)
│   ├── nginx.conf  Dockerfile
└── docker-compose.yml
```

## References

1. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014.
2. Pathak et al., *Context Encoders: Feature Learning by Inpainting*, CVPR 2016.
3. Iizuka et al., *Globally and Locally Consistent Image Completion*, SIGGRAPH 2017.
4. Yu et al., *Free-Form Image Inpainting with Gated Convolution*, ICCV 2019.
5. Barnes et al., *PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing*, SIGGRAPH 2009.
6. Wexler, Shechtman & Irani, *Space-Time Completion of Video*, IEEE TPAMI 2007.
7. Criminisi, Pérez & Toyama, *Region Filling and Object Removal by Exemplar-Based Image Inpainting*, IEEE TIP 2004.
8. Pérez, Gangnet & Blake, *Poisson Image Editing*, SIGGRAPH 2003.
9. Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa)*, WACV 2022.
10. Telea, *An Image Inpainting Technique Based on the Fast Marching Method*, J. Graphics Tools 2004.
