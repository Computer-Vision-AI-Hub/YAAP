# YAAP — Architecture & Code Documentation

YAAP (Yet Another Annotation Platform) is a local, self-hosted image annotation
tool developed by ComputerVisionAIHub ( one FastAPI
process, one SQLite file, static HTML/CSS/vanilla-JS frontend, and Ultralytics for every model-backed feature). Everything — images, labels, model weights, training runs —
are stored in `./data` on the host, bind-mounted into the container.

This document explains the backend's routers/services/db
layers, the frontend's view-router and canvas editor, the Docker packaging, and the
on-disk data layout, plus the end-to-end flows that tie them together (annotate → QC →
generate → train, and SAM-assisted segmentation).

---

## 1. High-level architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│   Browser (vanilla JS)      │  HTTP  │           FastAPI (uvicorn)           │
│                             │◄──────►│                                        │
│  index.html                 │  JSON  │  app.py — mounts routers + /static     │
│  ├─ api.js   (fetch client) │        │                                        │
│  ├─ editor.js (canvas)      │        │  routers/  — one file per resource     │
│  └─ app.js   (hash router,  │        │  services/ — business logic, no HTTP   │
│      all views)             │        │  db.py     — SQLAlchemy models         │
│  yaap.css (design tokens)   │        │  schemas.py — Pydantic I/O contracts   │
└─────────────────────────────┘        └──────────────┬─────────────────────────┘
                                                        │
                                          ┌─────────────┴─────────────┐
                                          │     ./data  (bind mount)   │
                                          │  yaap.db          (SQLite) │
                                          │  projects/<id>/…   images  │
                                          │  models/…         weights  │
                                          │  runs/…      training logs │
                                          └────────────────────────────┘
                                          ┌────────────────────────────┐
                                          │  ./seg_models (bind mount) │
                                          │  SAM weights, checked      │
                                          │  before data/models/       │
                                          └────────────────────────────┘
```

There is no build step anywhere. The frontend is served as static files by FastAPI's
`StaticFiles` (`/static/...`); editing a `.js`/`.css` file and refreshing the browser is
the entire dev loop. Schema changes need no manual step either: on every startup
`init_db()` creates missing tables, adds columns the models have gained, and runs any
pending one-off migrations (§3.3).

---

## 2. Directory layout

```
YAAP/
├─ backend/
│  ├─ app.py               FastAPI app: router registration + static mount + "/"
│  ├─ config.py            All paths (DATA_DIR, PROJECTS_DIR, MODELS_DIR, SEG_MODELS_DIR)
│  ├─ db.py                SQLAlchemy engine/session + all table models
│  ├─ schemas.py           Pydantic request bodies (validation only, no ORM coupling)
│  ├─ routers/             One FastAPI router per resource — thin HTTP glue
│  │  ├─ projects.py       projects + classes CRUD
│  │  ├─ images.py         image upload/list/file/thumb/patch/delete
│  │  ├─ annotations.py    per-image annotation get/replace (editor autosave)
│  │  ├─ models.py         pretrained/weights listing, .pt upload, autolabel, active-learning
│  │  │                    ranking, Test model, idle-shutdown signals (/system/leaving|here)
│  │  ├─ sam.py            SAM prompt/exemplar endpoint, warm-up, dataset "find all"
│  │  ├─ datasets.py       augment preview, raw export, import, merge, versions, version health
│  │  ├─ training.py       training job start/list/log/stop/delete + SSH settings/test
│  │  └─ bootstrap.py      Bootstrap rounds: state, start, progress, stop
│  └─ services/            Business logic — no FastAPI/HTTP imports, all pure Python
│     ├─ storage.py        Project folder layout, upload dedup, thumbnailing
│     ├─ sam.py            SAM 3 loading + GPU sharing, point/box/exemplar/text prediction → polygons
│     ├─ inference.py      YOLO/RF-DETR loading + prediction → YAAP annotation dicts, uncertainty
│     ├─ augmenter.py      albumentations pipeline builders (preprocess + augment)
│     ├─ exporter.py       Dataset version generation (split/preprocess/augment/zip)
│                          + export_raw() — no-version, no-split zip pack
│     ├─ importer.py       Zip → project (import_zip) and project → project (merge_project)
│     ├─ health.py         Version health statistics (counts, balance, sizes, duplicates)
│     ├─ trainer.py        Launches/monitors/stops the training subprocess
│     ├─ train_worker.py   Standalone script run as that subprocess (actual model.train())
│     ├─ ssh_runner.py     SSH training: settings, copy, remote launch, log streaming, kill
│     ├─ remote_train.py   Standalone script copied to and run on the SSH host
│     ├─ bootstrap.py      Bootstrap: train on human-checked images, label the rest
│     ├─ jobs.py           Background-job plumbing: safe writes (claim_image), idle shutdown
│     └─ cleanup.py        Deleting a project and everything it owns
│
├─ frontend/
│  ├─ index.html           Shell: topbar, rail nav, <main id="stage">, toast host, <dialog>, watermark
│  ├─ css/yaap.css          Design tokens (CSS custom properties) + all component styles
│  ├─ img/                  yaap-logo.png (256²) + favicon.png (64²), generated from YAAP_logo.png
│  └─ js/
│     ├─ api.js            Thin fetch wrappers, one function per endpoint
│     ├─ editor.js          Editor class: canvas drawing, tools, pan/zoom, SAM prompt state
│     └─ app.js             Hash router + every view (one view = one function)
│
├─ tests/                   pytest backend tests + tests/frontend/ headless-Chrome UI tests (see TESTS.md)
├─ data/                    Bind-mounted; DB + per-project images/thumbs/versions
├─ seg_models/               Bind-mounted; dedicated SAM weights directory
├─ FEATURES_USAGE.md        How to use and manually test each newer feature
├─ TESTS.md                 What every automated test checks and how to run them
├─ YAAP_logo.png             Source logo (full-res) — resize into frontend/img/ if it changes
├─ Dockerfile               cpu + gpu build targets, non-root "yaap" user
├─ docker-compose.yml       yaap (gpu) + yaap-cpu services, volumes, PRELOAD_MODELS
├─ entrypoint.sh            Runs as root: chown bind mounts, then drops to "yaap" user
├─ start.sh                 Runs as "yaap": optional weight preload, then uvicorn
├─ requirements.txt         Pinned/loose deps (torch installed separately for CPU/GPU wheel)
└─ run.py                   Bare-metal entry point: `python run.py` → 127.0.0.1:8811
```

---

## 3. Backend

### 3.1 `app.py` — composition root

Creates the `FastAPI()` instance, calls `init_db()` (creates/upgrades the schema, §3.3) and
`jobs.fail_interrupted_jobs()` (marks auto-label/find-all/ranking jobs the previous server
was running as failed — they ran in-process and died with it), registers every router
(including `training.ssh_router` and `bootstrap.router`), mounts `frontend/` at `/static`
through `NoCacheStatic` (sends `Cache-Control: no-cache`, so browsers revalidate UI files
after an update), and serves `frontend/index.html` at `/`.
That's the entire file — no middleware, no auth, no CORS (this is a single-user,
localhost-only tool by design; `docker-compose.yml` binds `127.0.0.1:8811` specifically to
keep it off the network).

### 3.2 `config.py` — paths, single source of truth

```python
BASE_DIR = .../YAAP
DATA_DIR = BASE_DIR/data
PROJECTS_DIR = DATA_DIR/projects
MODELS_DIR = DATA_DIR/models          # uploaded .pt + ultralytics' own download cache
RUNS_DIR = DATA_DIR/runs              # training logs + ultralytics run folders
DB_PATH = DATA_DIR/yaap.db
SEG_MODELS_DIR = env SEG_MODELS_DIR or BASE_DIR/seg_models   # dedicated SAM weights dir
```
All five directories are created eagerly at import time (`mkdir(parents=True,
exist_ok=True)`). `PRETRAINED_MODELS` is a dict of task → list of Ultralytics aliases shown
in the Auto-label/Train dropdowns; `RFDETR_VARIANTS` lists the two optional RF-DETR sizes
(needs the separate `rfdetr` pip package); `RTDETR_VARIANTS` lists the two RT-DETR sizes,
which need nothing extra — `ultralytics.RTDETR` ships inside the `ultralytics` package
already required for YOLO, and is always available.

### 3.3 `db.py` — the whole schema, eleven tables

SQLite via SQLAlchemy 2.0, one global `engine`/`SessionLocal`. `get_db()` is the FastAPI
dependency every router uses (`Depends(get_db)`), a generator yielding a `Session` and
closing it after the request. Every connection runs in **WAL** mode with a 30 s busy timeout
(`_sqlite_pragmas`): readers (job polling, image lists) never block a writer's commit, and a
writer waits for another instead of failing with "database is locked" after sqlite3's default
5 s — background jobs, the editor and Review write at the same time. The find-all jobs also
skip (and log) a single image whose write still fails, instead of ending the whole job.

| Table | Purpose | Notable columns |
|---|---|---|
| `Project` | one dataset/task | `task_type`: `detect`\|`segment`\|`classify` — fixed at creation, drives every downstream tool choice |
| `LabelClass` | a class within a project | `order_idx` (becomes the YOLO class index on export — **keep stable once training starts**), `color` |
| `ImageAsset` | one uploaded image | `status`: `unannotated`\|`annotated`\|`auto-annotated`\|`review`\|`needs_reannotation`\|`empty`\|`approved` (see §6.3); `split`: `""`\|`train`\|`val`\|`test` — a **manual pin** that `_split_images()` respects before randomizing the rest |
| `Annotation` | one shape on one image | `kind`: `bbox`\|`polygon`\|`classification`; `data` is a JSON string (`{x,y,w,h}` for bbox, `{points:[[x,y]…]}` for polygon, `{}` for classification); `source`: `manual`\|`model`; `confidence`; `origin`: which tool made a model annotation (`sam`\|`autolabel`\|`bootstrap`, `""` = a person or older rows) |
| `DatasetVersion` | a frozen, generated snapshot | `config` (the full `VersionCreate` body, JSON), `stats` (output of `exporter.generate()`), `status`: `building`\|`ready`\|`failed` |
| `ModelWeight` | an uploaded or trained `.pt` | `project_id` nullable = global weight available to all projects; `source`: `uploaded`\|`trained` |
| `TrainJob` | one training run | `pid` (OS process id of the local worker, used to `stop()`/`refresh_status()`), `status`: `queued`\|`running`\|`done`\|`failed`\|`stopped`, `weights_path` (best.pt once done), `params` (JSON: hyperparameters, plus `mode:"ssh"` + a settings snapshot for SSH runs, or `bootstrap_run`/`bootstrap_round` for Bootstrap — those have `version_id=0`, they train on their own temporary set) |
| `AutolabelJob` | one background job: auto-label, SAM find-all, generic find-all, uncertainty ranking, or a Bootstrap labeling pass | `status`: `running`\|`done`\|`failed`, `total`/`processed`/`images_labeled`/`annotations_added`, `log`, `skipped` and `labeled_image_ids` (JSON lists), `error` |
| `ImageScore` | active-learning uncertainty of one image | `image_id` (unique), `uncertainty` 0 (sure) … 1 (torn), `model_name`; deleted with its image |
| `Proposal` | a class-less object outline from generic "find all", waiting for a class | `data` = `{"polygon": [...], "bbox": {...}}`, `score`; becomes a normal `Annotation` once labeled in the editor; deleted with its image |
| `BootstrapRun` | one Bootstrap round (§3.6 `bootstrap.py`) | `round`, `status`: `training`\|`labeling`\|`done`\|`failed`\|`stopped`, `config` (request + the class-name → id map at start), `train_job_id`, `autolabel_job_id`, `weight_id`, `epoch`/`epochs`, `train_images`/`target_images`, `message`, `error` |

`Annotation.data_obj` is a convenience property that lazily `json.loads`es `data`.

**Schema upgrades.** `init_db()` (run by the server at startup and by every training
worker) brings any existing `yaap.db` up to date in one transaction:
1. `create_all()` creates tables that don't exist yet;
2. `_add_missing_columns()` compares each model with the table on disk and runs
   `ALTER TABLE … ADD COLUMN` for every column a model has gained — rows that already
   exist get the column's plain default (`""`, `0`, …; computed defaults such as `now()`
   can't be expressed, those rows get `NULL`). A new primary-key or unique column can't be
   added this way; that raises a clear error asking for a migration;
3. missing indexes are created (`create_all` only indexes brand-new tables);
4. `MIGRATIONS` — an ordered list of functions for everything ALTER TABLE can't say (data
   fixes, renames, constraints) — runs the entries this database hasn't seen. SQLite's
   `PRAGMA user_version` stores how many have run; a brand-new database skips them all
   (it is already in the final shape). Append new entries, never edit or reorder old ones.
   `_m1_reset_empty_predictions` is the first: it resets images that an old auto-label
   bug left `auto-annotated` with no annotations.

### 3.4 `schemas.py` — request bodies only

Pure Pydantic models used for **input validation**, never returned directly (every router
hand-rolls its own `serialize_*()` dict instead — see below). Notable ones:
- `VersionCreate.approved_only` defaults to `True` — Generate refuses to run unless
  images are QC-approved (or you explicitly uncheck the box in the UI).
- `MergeRequest` — just `source_project_id` + optional `default_split`.
- `TrainRequest.extra` — an escape hatch dict merged straight into `model.train(**extra)`
  in `train_worker.py`, for any Ultralytics kwarg not exposed in the UI.
  `TrainRequest.mode` is `local` or `ssh`; `SshSettings` is the SSH form (no password
  field — key auth only).
- `InferenceRequest.create_missing_classes` defaults to `False`: a model's classes that the
  project doesn't have are dropped instead of being added to the class list.
- `ImagePatch.expect_annotations` — optional list of annotation ids; the PATCH answers
  409 if the image's annotations aren't exactly those (Quick Review's Accept sends it).
- `BootstrapRequest` — base model (empty = `yolo11n` for the task), `epochs` (0 = auto),
  `imgsz`, `conf`, `iou`, `device`, `refresh_unreviewed`.
- The SAM requests (`SamRequest`, `FindAllRequest`, `WarmupRequest`) live next to their
  endpoints in `routers/sam.py`.

### 3.5 Routers — thin HTTP layer

Every router follows the same shape: parse path params, load the row (`project_or_404` /
`_img` helpers), delegate to a service function, catch the service's domain exception
(`RuntimeError`, `ValueError`, `SamError`, `InferenceError`) and turn it into an
`HTTPException` with a human-readable message, return a plain dict (FastAPI JSON-encodes
it). None of the routers talk to the ORM beyond simple lookups — all real logic lives in
`services/`.

- **`projects.py`** — also defines `PALETTE` (10 hex colors auto-assigned to new classes)
  and the shared `project_or_404()` helper imported by almost every other router.
  `serialize_projects()` computes the counts on the fly in a fixed number of SQL queries:
  `annotated_count` = images a person has done or checked (status `annotated`, `approved`
  or `empty`), `predicted_count` = model output nobody has reviewed yet (`auto-annotated`,
  `review`); rejected images count as neither. Deleting a project goes through
  `cleanup.delete_project()` (§3.6), not a bare ORM delete.
- **`images.py`** — the image list (`GET …/images`) counts annotations, the lowest model confidence, proposals and
  uncertainty in grouped queries for the whole project (0.2 s for 5 000 images; per-image loads took 6 s). Upload,
  zip import and weight upload are plain `def` endpoints (FastAPI runs them in worker threads): as `async def` their
  synchronous work froze every other request until they finished; the import reads the zip from the upload's
  temporary file instead of memory (5 000 images: 391 MiB instead of 1.4 GB).
  Upload accepts a `list[UploadFile]`; unsupported extensions go to
  `skipped` rather than erroring the whole batch. `_img()` is the per-image
  ownership-check helper reused by `annotations.py` and `sam.py`. `serialize_image()` also
  reports `min_conf` (lowest model confidence on the image), `uncertainty` (active-learning
  score) and `proposal_count`; `GET`/`DELETE /{iid}/proposals` serve the generic find-all
  outlines. `PATCH` with `expect_annotations` first takes SQLite's write lock (a no-op
  UPDATE), then compares the annotation ids, so a background job can't swap an image's
  predictions between the check and the status change.
- **`annotations.py`** — a single `PUT` that **replaces the entire annotation set** for
  an image (delete-all-then-insert). This is what the editor's autosave calls every time
  you stop drawing for 700ms (see `markDirty()`/`save()` in `app.js`). Also flips
  `image.status` between `annotated`/`unannotated` based on whether the new set is empty.
  Geometry is validated (`AnnotationIn`: a bbox needs finite x/y and w, h > 0, a polygon ≥ 3 points → 422), and the
  kind must fit the project: a classify image holds at most one `classification` tag, detect/segment images none.
  With `expect_ids` (the ids the editor last loaded/saved) the PUT takes the write lock and answers **409**
  if the image now holds other annotations — a job wrote predictions or another tab saved — and the editor
  merges instead of silently deleting them. Annotations of a class deleted meanwhile are dropped (reported as
  `dropped_unknown_class`) instead of refusing the whole save. The response lists the new annotations in the
  order they were sent, so the editor can take over their (new) ids.
- **`models.py`** — `/api/models` lists three sources merged into one dropdown: DB-stored
  `ModelWeight` rows (project-specific + global), `PRETRAINED_MODELS` aliases (Ultralytics
  downloads these on first use), and RF-DETR variants if the `rfdetr` package is
  installed. `/autolabel` is the heaviest endpoint, and runs as a background job (an
  `AutolabelJob` row tracks `status`/`processed`/`total`/`log`, polled by the UI) rather
  than blocking the request — for each image it calls `inference.predict()`, optionally
  auto-creates missing classes (matched **by name**, same pattern reused in
  `importer.py`), and marks touched images `status="auto-annotated"` (not `"review"` —
  that would skip human review entirely). `/autolabel/{jid}/preview` returns exactly the
  images that job touched, with their new annotations, for the preview-grid the UI shows
  once the job finishes; approving there (per-image or in bulk) is what actually promotes
  an image to `status="review"`, via the ordinary `PATCH .../images/{iid}` endpoint.
  Three safety rules in `_run_autolabel()`: new boxes go through `jobs.drop_duplicates()` (one box per
  object, an existing box always wins — `InferenceRequest.max_overlap`); an image is written only if its status is still
  what the job saw when it started (`jobs.claim_image()`, §3.6) — so an image you label,
  approve or reject meanwhile is left alone — and **"replace existing" deletes old
  annotations only when there are new ones to put in their place**; if the model finds
  nothing (or only classes the project doesn't have) the image keeps its annotations and
  status. `POST /projects/{pid}/active-learning/score` runs the uncertainty ranking as
  another `AutolabelJob` (writes `ImageScore` rows, no annotations) over every image nobody
  has checked yet — unannotated, or holding only unreviewed predictions (`auto-annotated`,
  `review`), which after a Bootstrap round or find-all is almost all of them. `GET
  /projects/{pid}/active-learning` reports its state. `/system/leaving` and `/system/here`
  are the idle-shutdown handshake (§3.6 `jobs.py`).
- **`sam.py`** — `GET /api/sam/models` (static capability list), `POST /api/sam/warmup`
  (loads both SAM 3 copies ahead of the first click), and `POST
  /projects/{pid}/images/{iid}/sam`: a `point`/`box` prompt → candidate masks, or an
  `exemplar` box → every similar object in that image (the editor's `E`). Stateless — the
  frontend accumulates click history and resends the full point/box set on every prompt.
  `POST /projects/{pid}/sam/find-all` runs over the dataset as an `AutolabelJob`: in
  `classes` mode each ticked class has a text phrase, matches become `model` annotations
  (`auto-annotated`), the same object hit by two phrases is kept once (higher score wins)
  and objects already annotated with that class are skipped; in `generic` mode generic
  nouns outline every object and the results are stored as class-less `Proposal` rows.
  Like auto-label, it writes an image only if nobody changed it meanwhile.
- **`datasets.py`** — the biggest router: augmentation preview (single contact-sheet
  JPEG, no zip), the "just pack what I have" raw export, dataset import, project-to-project
  merge, and the versioned Generate pipeline (`POST /versions` kicks off
  `_build_version()` as a FastAPI `BackgroundTasks` job — the HTTP response returns
  immediately with `status="building"`, and the frontend polls `GET /versions` every 3s
  until it flips to `ready`/`failed`). `GET /versions/{vid}/health` computes the health
  statistics from the exported files (`services/health.py`) and caches them in the
  version's `stats`; `?refresh=true` recomputes.
- **`training.py`** — starts `trainer.start()`, which spawns a **separate OS process**
  (not a background task) so a crash or `SIGTERM` can't take the web server down with it,
  and so `stop()` can kill it by PID group. Jobs report `mode` (`local`/`ssh`),
  `ssh_host` and `bootstrap_round`; `DELETE /train/{jid}?remote=true` also removes an SSH
  job's folder on the remote machine. A second router, `ssh_router`
  (`/api/training/ssh/settings` GET/PUT, `/api/training/ssh/test` POST), holds the SSH
  settings and the connection test.
- **`bootstrap.py`** — `GET /projects/{pid}/bootstrap` (counts for the card + the last
  round), `POST` (start a round; 400 with a readable reason if it can't, 409 while another
  round runs), `GET /{bid}` (progress, with the labeling `AutolabelJob` nested and
  `to_review` = this round's images still waiting in Review), `POST /{bid}/stop`.

### 3.6 Services — the actual logic

- **`storage.py`** — defines the on-disk layout (`data/projects/<pid>/{images,thumbs,versions}`)
  and the one function everything else calls to persist an upload: `save_upload()`.
  It first checks that the bytes are a readable image (nothing is written for a corrupt
  file), de-duplicates filenames by appending `_1`, `_2`, …, stores an EXIF-rotated photo
  **upright** (re-encoded once, without the orientation tag, so rotation metadata never has to
  be dealt with downstream — RF-DETR's PIL loader included), stores formats a browser can't
  display (TIFF, 16-bit/float images) as 8-bit PNG under `<stem>.png`, and builds a 320px JPEG
  thumbnail. Both `importer.py` and the plain upload endpoint funnel
  through this single function, so dedup/thumbnailing behavior is consistent everywhere.

- **`sam.py`** *(service)* — SAM 3 in two forms: ultralytics' `SAM` for click/box
  prompts (`predict()`), and `SAM3SemanticPredictor` for "find all" — a box exemplar
  (`predict_all()`) or text phrases (`predict_phrases()`, which encodes the image once
  for all phrases; this needs ultralytics' CLIP fork, see §5). Weights resolve from
  `SEG_MODELS_DIR` before `MODELS_DIR`, and only names listed in `SAM_MODELS` are ever
  loaded (a `.pt` is a pickle). **GPU sharing:** each copy takes ~2 GB, so only one lives
  on the GPU — `_use(kind)` moves the requested one there and parks the other in CPU RAM
  (`_park()`, ~1 s to swap back vs ~8 s cold), all under one `_gpu_lock` so a request
  can't park a model another request is using. `warmup()` loads both ahead of time,
  `release_gpu()` parks everything (Bootstrap calls it before training). Click/box prompts reuse the image's
  encoding (`_prompt_cached()`): the first prompt on an image encodes it (~2.2 s for a phone photo on an RTX 4050),
  every further click on that image takes ~0.03 s (≈0.2 s through the API instead of ~2.7 s); the encoding is
  dropped when the image or its file changes, on warm-up and when the model is parked. "Find all" decodes
  a copy downscaled to 1280 px and runs SAM 3 at its native **1008 px** input (`FIND_ALL_IMGSZ`;
  without an explicit `imgsz` ultralytics used its generic 640 → 644, which lost small objects:
  on VisDrone 1008 raised F1 .71 → .75 and tiny-object recall .32 → .73); on CUDA out-of-memory it
  retries once at 644, then fails with a clear message. Masks become polygons in `mask_to_polygons()`
  (`cv2.findContours` + `approxPolyDP`, anything under `min_area=60` dropped).

- **`inference.py`** — the auto-label counterpart to `sam.py`: lazy-cached YOLO/RT-DETR/
  RF-DETR loading, `resolve_weights()` turns a `"weight:<id>"` string into a real path (or
  passes ultralytics aliases straight through, letting Ultralytics download them on demand
  into `MODELS_DIR` via `YOLO_CONFIG_DIR`). `predict()` branches on `task` (classify →
  top-1 label, segment → polygon masks, detect → boxes) and always returns
  `{class_name, kind, data, confidence}` dicts — i.e. it speaks in the same shape as
  hand-drawn annotations, which is exactly why `autolabel()` in `models.py` can push its
  output straight into the `Annotation` table. `RTDETR` results share the exact same
  `.boxes`/`.names` structure as `YOLO`'s (both inherit ultralytics' common `Results`
  class), so `is_rtdetr_arch()` only decides *which loader* to call
  (`_load_rtdetr` vs `_load_yolo`) — every line downstream of that is unaware which one
  produced the result. The same detection happens a second time in `trainer.py`, *before*
  `resolve_weights()` resolves a `"weight:<id>"` reference to its file path, because a
  resolved path for a continued/fine-tuned weight won't contain "rtdetr" in the string —
  the family is instead threaded through as `params["is_rtdetr"]` in the `TrainJob` row,
  and `train_worker.py` picks `RTDETR` vs `YOLO` from that flag rather than guessing from
  the path. When training finishes, the new `ModelWeight.arch` is tagged
  `"rtdetr:<file>"` for rtdetr runs specifically, so a *future* `is_rtdetr_arch()` call on
  that trained weight (e.g. auto-labeling with it later) still detects the family
  correctly — everywhere else `arch` keeps its original untagged value, so this changes
  nothing for existing YOLO weights. `uncertainty()` scores one image for active learning:
  detection/segmentation use `detection_uncertainty()` (1 − the mean confidence of the
  detections ≥ 0.25, nothing found = 0.5 — the earlier "most ambiguous detection decides"
  saturated on busy images and didn't correlate with model errors);
  classification uses the normalized entropy of the class probabilities. Segment predictions are
  simplified with `simplify_polygon()` (Douglas-Peucker, 0.008 × perimeter — the same as SAM's outlines),
  so a mask outline of hundreds of points becomes an editable polygon of ~10.

- **`augmenter.py`** — wraps `albumentations`. `build_preprocess()` handles resize
  (stretch or letterbox-pad) and grayscale; `build_augment()` maps every UI slider/toggle
  (hflip, rotate, shear, brightness/contrast, hue/sat, blur, noise, CLAHE, cutout) onto an
  albumentations transform with UI-value-derived limits. The tricky part is
  `apply()`: bboxes go through `BboxParams`, but **polygons are routed through
  `KeypointParams`**, each vertex tagged with a scalar id (`ann_idx * 100000 +
  point_idx`) so they can be regrouped back into polygons after the transform —
  albumentations has no native "polygon" primitive, only bboxes and keypoints, so this
  encoding is what lets rotation/shear/flip keep segmentation masks geometrically valid.
  Polygons that collapse to a degenerate shape after cropping are dropped.

- **`exporter.py`** — two entry points:
  - `generate(db, project, version)` — the full "Generate dataset version" pipeline:
    filter images (annotated / approved-only per config), split into train/val/test
    (`_split_images()`, which **respects `ImageAsset.split` pins first**, then randomizes
    the rest with a fixed seed for reproducibility), preprocess every image once, and for
    the train split only, additionally generate `multiplier - 1` augmented variants
    (**val/test never get the augment pipeline** — only `build_preprocess`). The
    preprocessing/augmentation pipeline runs identically regardless of output format.
    detect/segment tasks always get the `images/`+`labels/` per-split layout plus a
    `data.yaml` — the layout YOLO/RT-DETR training via ultralytics expects, written
    unconditionally so every version is trainable in-platform regardless of
    `config["format"]`. classify tasks always use class-per-folder instead (no format
    choice — `generate()` raises early if `format == "coco"` for a classify project).
    `config["format"] == "coco"` names the validation folder `valid/` (Roboflow's convention — RF-DETR reads
    `valid/_annotations.coco.json`; `data.yaml` then says `val: valid/images`) and additionally writes, per split, one flat copy of the
    images alongside a single `_annotations.coco.json` (`_write_coco_json()`) —
    matching the Roboflow-style layout that pycocotools, HuggingFace `transformers`,
    detectron2, and Roboflow's own `rfdetr` training script all expect out of the box.
    Bbox annotations get `segmentation: []`; polygon annotations get a flattened
    single-ring `segmentation` plus their enclosing rectangle as `bbox`
    (`_coco_ann()`). Category ids are 1-based, matching COCO convention. This is a
    second, parallel layout, not a replacement — a `"coco"` version's `data.yaml` is
    identical to a `"yolo"` version's, so `"coco"` is strictly additive: images end up
    written twice (once under `<split>/images/`, once flat under `<split>/`) since the
    two layouts disagree on where the image sits.

    Either way, zips the whole version folder and returns stats including `images` (files
    written, augmented copies included), `source_images`/`source_images_total` (distinct
    images per split, used by the Versions view to show real percentages), `format`,
    `split_ids` (which image went to which split — Test model reads its test images from here,
    not by re-splitting the project's current images), `skipped` (images or augmented copies
    that could not be written, with the reason — the Versions list shows a ⚠ pill) and
    `bad_annotations` (broken shapes left out; one used to drop its whole image silently) and `tiny_dropped`
    (labels smaller than 4 px² after the resize, which albumentations removes — e.g. 4-px objects in a 2000-px photo
    resized to 640; shown in the Versions list).
    `data.yaml` has no `path:` key, so ultralytics resolves the splits next to the file and a
    downloaded zip trains as-is on another machine.
  - `export_raw(project, approved_only)` — the "just give me a zip" path: no
    `DatasetVersion` row, no split, no preprocessing, no augmentation, **YOLO format
    only** (no format choice here — this is the quick-pack path, not the versioned one).
    Copies images as-is into `images/`+`labels/` (or class folders; a second image with the
    same stem, e.g. `x.jpg` + `x.png`, becomes `x_<id>` so labels never collide), writes
    `classes.txt` (also for classification, so a re-import keeps the class order) + a minimal `data.yaml`, zips —
    each request in its own temporary folder
    (two downloads at once used to delete each other's files), removed after the response
    is sent (`remove_export`) — rebuilt fresh on every request so it always reflects
    current annotations.

- **`importer.py`** — the inverse of `exporter.py`, plus project-to-project merge:
  - `import_zip()` sniffs a `data.yaml`/`classes.txt` for class names, then treats **any**
    image file in the zip as an entry, matched to its label file by path first
    (`…/images/x.jpg` → `…/labels/x.txt`, then a `x.txt` next to the image, then any `x.txt` —
    the one sharing the most folders — so both `images/x.jpg`+`labels/x.txt` and nested
    `train/images/x.jpg`+`train/labels/x.txt` layouts work, and `train/x.jpg` never gets
    `valid/x.txt`), pins `image.split` from any `train`/`val`/`test` path component it finds. A
    malformed label line is skipped and reported (`label_lines_skipped`, `label_line_errors`)
    instead of aborting a half-done import. Classes are
    matched **by name** against the target project — same helper
    (`_get_or_create_class`) as autolabel uses, so imports and model predictions
    "just work" against an existing class list without creating duplicates.
  - `merge_project(target, source)` — copies every image file (`storage.save_upload`
    handles the physical copy + rename-on-collision + rethumbnail), maps each source
    annotation's class **by name** into the target project (creating it there if it
    doesn't already exist), and refuses to run across mismatched `task_type`s.

- **`trainer.py`** / **`train_worker.py`** — `trainer.start()` resolves the starting
  weights, writes a `TrainJob` row, and `Popen`s
  `python -m backend.services.train_worker --job <id>` with `start_new_session=True`
  (its own process group, so `stop()` can `os.killpg()` the whole tree including any
  DataLoader worker processes). `train_worker.py` is a tiny standalone script — it opens
  its own DB session, loads the job's params, calls `YOLO(...).train(...)`, and on
  success registers `best.pt` as a new global `ModelWeight` (so it immediately shows up
  in Auto-label/Train dropdowns for the next round — the classic "annotate → train →
  auto-label → review → retrain" loop this whole platform is built around).
  `refresh_status()` is a defensive check called on every job-list/log request: if the
  DB still says `running` but the worker is gone, it flips the job to `failed` without
  ever having been told explicitly (covers the container-restarted-mid-training case).
  "Gone" is decided by `worker_alive()`: it reaps a finished child (`waitpid`), then reads
  `/proc/<pid>/cmdline` and only accepts the pid if it really is
  `train_worker --job <this id>` — after a container restart pids start over, and a stored
  pid can belong to any process, even the server. `kill_worker()` uses the same check
  before `killpg`. `launch()` (split out of `start()`) spawns the worker for any inserted
  `TrainJob`; Bootstrap uses it too. `stop()`/`delete()` on a Bootstrap job stop the whole
  round. `train_worker.py` registers the weight under `params["weight_name"]` when given,
  runs Bootstrap's labeling pass after training when `params["bootstrap_run"]` is set,
  and calls `jobs.job_finished()` on exit. SSH jobs (`params["mode"] == "ssh"`) go
  through `run_ssh()` → `ssh_runner.run_job()` instead.

- **`ssh_runner.py`** / **`remote_train.py`** — training on another machine over SSH,
  key authentication only (`BatchMode=yes`, so ssh can never stop and ask for a
  password; `StrictHostKeyChecking=accept-new` with YAAP's own `data/ssh_known_hosts`).
  Settings live in `data/ssh.json`; host/user are validated against option injection
  (`-oProxyCommand=…`). A job: copy the version folder and `remote_train.py` (rsync,
  falling back to scp), start `nohup setsid sh -c '…'` remotely (it writes `pid` and
  `exitcode` files), stream `train.log` with `tail -F` (resuming from the byte offset
  after a dropped connection, giving up after 15 minutes unreachable), then copy the run
  folder back to `runs/job_<id>/` and register `best.pt`. Stop kills the remote process
  group; delete can remove the remote `job_<n>` folder (refuses any other path).
  `remote_train.py` is standalone (it can't import `backend`): it rewrites `data.yaml`'s
  `path:` for the remote folder and prints `YAAP_RESULT {"best": …}`.

- **`health.py`** — `compute(vdir, task)` reads an exported version folder and returns
  images per split, empty images, labels per class and split, imbalance ratio,
  objects-per-image histogram, object size and aspect statistics, duplicate images by MD5
  (within a split and across splits — the latter is leakage), warnings and a 0–100 score.

- **`bootstrap.py`** — one round = train a small YOLO model on every human-checked image
  (`annotated`/`approved`, plus `empty` ones as background), then label the targets
  (`unannotated` images; with `refresh_unreviewed` auto-annotated ones and with `relabel_rejected`
  `needs_reannotation` ones — in both cases only if every box on them came from a model)
  → `auto-annotated`. `start()` validates (≥5 labeled images; classify
  needs two classes), builds the training set under `runs/job_<id>/dataset/` — **copies,
  never hard links**, because ultralytics re-saves JPEGs it thinks are corrupt in place —
  with validation = the training images, picks epochs (`auto_epochs`: 300 for ≤15 images,
  150 for 30, down to 50 for 90+ — on VisDrone the old 100–150 left small-object models
  under-trained, F1 .38 vs .51) and the batch (`auto_batch`: 8, or 16 from 64 images, halved
  for every doubling of imgsz above 640), stores the class-name → id map, and launches the worker.
  Every annotation a model writes records its tool in `Annotation.origin` (`sam`, `autolabel`,
  `bootstrap`); a round only replaces its own earlier guesses (and pre-origin rows), and keeps
  predictions made by SAM / Run a model unless `relabel_other_tools` is set — an early round is
  often worse than SAM. In the
  worker, `track_epochs()` mirrors progress into the row and `label_phase()` deletes the
  copied set, then labels image by image through `_save()`: it writes only if the
  image's status is unchanged since the round picked it (`jobs.claim_image`); a new
  guess replaces an unreviewed old one, and "nothing found" keeps the old one. One failing
  image is skipped; if all fail the round fails. `refresh()` fails a round whose worker
  died silently (with a grace period while the worker is still being launched), `stop()`
  kills it.

- **`jobs.py`** — shared by every background job. `drop_duplicates(new, existing, max_overlap)` keeps
  one box per object for every model-based labeling path (auto-label, Bootstrap, both find-all modes):
  candidates are tried best-score first; one is dropped if it overlaps a kept or already-present box by
  IoU ≥ 0.5 with the same class (`SAME_CLASS_IOU`, as in NMS) or by IoU ≥ `max_overlap` (default 0.7)
  with any class; existing annotations always win; whole-image labels allow one per image. Items come
  from `shape()`: polygons are compared by area (`poly_iou`, both filled into a 256-px raster — two
  triangles splitting a square share a box but not an object), everything else by box. `claim_image(db, iid, seen)` runs a
  no-op `UPDATE … WHERE status = seen`: it takes SQLite's write lock and tells whether a
  person changed the image since the job picked it, so the check and the job's writes
  until commit are one atomic step. The idle-shutdown handshake: `request_leave()`
  (behind `/system/leaving`, sent when the tab closes) writes `data/.leave_at` when
  nothing runs, or `data/.leave_pending` while a training job, Bootstrap round or
  auto-label/find-all/ranking job runs; `job_finished()` (called when each of those
  ends) turns a pending leave into `.leave_at` once nothing is busy; `cancel_leave()`
  (behind `/system/here`) removes both. `fail_interrupted_jobs()` runs at startup and also
  drops a `.leave_pending` left by a job that died with the old server. Every tab sends
  `/system/here?tab=<id>` on load and every 20 s, and `/system/leaving?tab=<id>` on close:
  while another tab has been seen in the last 60 s, leaving arms nothing (closing one of two
  tabs used to stop the stack). `busy_reason()` refreshes training jobs first, so a dead
  worker doesn't keep a shutdown pending forever, and `trainer.refresh_status()` calls
  `job_finished()` when it marks one failed.

- **`cleanup.py`** — `delete_project()`: the ORM cascade only covers images (with their
  annotations, scores, proposals), classes and versions, so this also deletes the
  project's training jobs (stopping a running one or a Bootstrap round first, then
  `trainer.delete()` removes `runs/job_<id>/`, the log and the registered weight), the
  weights uploaded for the project (the file only if YAAP stored it under
  `data/models`/`data/runs` and no other weight row uses it), its auto-label jobs and
  Bootstrap rounds, then the project folder. Remote SSH folders are left alone.

---

## 4. Frontend

No framework, no bundler. Three script tags load in order (`api.js` → `editor.js` →
`app.js`), each attaching globals (`API`, `Editor`, and everything in `app.js`) that the
next one uses.

### 4.1 `index.html` — the shell

Fixed chrome that never gets replaced: a `.topbar` (brand — logo image + name +
subtitle — current-project breadcrumb + task badge injected by `app.js`, and a device
chip showing CPU/CUDA status), a `.rail` (the left nav — hidden until a project is
open), a `<main id="stage">` that every view function overwrites wholesale via
`innerHTML`, a toast host, and a single reusable `<dialog id="modal">` (used today only
for "new project").

A `.watermark` div (fixed, centered, `pointer-events:none`, `z-index:0`) sits behind
everything as the first element in `<body>` — two faint lowercase lines, "yet another
annotation platform" / "powered by ComputerVisionAIHub" (the second brighter than the
first). It only shows through empty space: `.topbar`/`.shell` were given
`position:relative;z-index:1` specifically so their own opaque backgrounds paint over
it rather than the watermark bleeding through solid panels. The brand logo
(`frontend/img/yaap-logo.png`, 256×256) and favicon (`frontend/img/favicon.png`, 64×64)
are both generated from `YAAP_logo.png` at the repo root via Pillow — regenerate them
with a one-off resize script any time that source file changes, there's no build step
wired up to do it automatically.

### 4.2 `api.js` — API client

One object, `API`, with one method per endpoint. `API._j()` is the shared JSON
fetch-and-throw helper — every other method is a one-liner calling it, except the ones
that need `FormData` (`upload`, `importDataset`, `uploadWeight`) or a raw blob
(`augPreview` returns an object URL). This file has no logic beyond "shape the request,"
which keeps `app.js` free of any `fetch()` calls at all. Errors are thrown as `Error`s
carrying the server's `detail` message and the HTTP status (`e.status`), so a view can
react to a specific code (Quick Review reloads the image on a 409).

### 4.3 `app.js` — router + every view

A tiny hash router: `#/` → `viewHome()` (project grid), `#/p/<id>/<view>` → one of the
view functions in the big dispatch table in `route()`. Global mutable state lives in one
object, `S` (current project, image list, current image index, the live `Editor`
instance, poll-interval handles). Each view function is self-contained: it sets
`$("#stage").innerHTML`, wires up event handlers, and does its own data fetching — there's
no shared component model, view functions just fully re-render on every navigation.

The topbar chip shows `N imgs · N labeled · N predicted` (see `annotated_count` /
`predicted_count`, §3.5); the project cards on Home show the same numbers.
`toast(msg, kind, ms)` is the shared notification (e.g. "✓ Uploaded N image(s)").

Views, in rail order:

- **`viewUpload`** — dropzone (drag or click-to-browse), the image grid, and the
  "Dataset in / out" card: raw zip download (`API.exportRawUrl` + a synthetic `<a
  href>.click()`), zip import, and the merge-from-another-project dropdown (only lists
  projects with a matching `task_type`).
- **`viewReview`** *(the QC screen)* — filter chips with live counts; each grid tile has
  an "open in editor," a quick-approve, and a quick-reject button (reject sets
  `status="needs_reannotation"`; click again to clear it — see §6.3 for why this is
  separate from auto-label's `review`). Clicking the tile itself also opens the editor
  (`openReviewImage` reassigns `S.images`/`S.curImage` and navigates to `annotate`, so
  Prev/Next inside the editor walks the same filtered ordering you were browsing). The
  default filter is a pseudo-status, `"reviewable"` (labeled "To review" in the UI) —
  `qcMatch()` defines it as `annotated`, `auto-annotated`, `empty` or `review`, i.e.
  *deliberately excluding* `unannotated` (nothing to review yet), `needs_reannotation`
  (waiting on a fix) and `approved` (already resolved). This is what makes the queue actually drain to empty as you
  approve/reject your way through it, rather than staying permanently populated — an
  earlier version defined it as "anything not unannotated," which included `approved`
  and meant reviewed images never left the list.
- **`viewClasses`** — name (contenteditable, saved on blur) + color picker + delete per
  row; the row order **is** the export class index, called out explicitly in the copy.
- **`viewQuickReview`** — one image at a time: **Accept** (`A`) / **Reject** (`R`) in
  `review` mode, prev/next + revert in `approved` mode. Accept sends the ids of the
  annotations on screen (`expect_annotations`); on a 409 it re-renders the image with
  the current ones. **⏏ Exit** / `Esc` return to the project Home. `S.reviewOnly` (one-shot) confines the queue to given image ids —
  Bootstrap's "Review round N" uses it. Classification annotations show as a class tag.
- **`viewAnnotate`** — the canvas editor screen; see §4.4/§4.5, this is the largest and
  most stateful view. With no images in the chosen scope it shows "No images to
  annotate" + Return to Home. **Most uncertain first** (shown once images have been
  ranked) walks unlabeled images by `uncertainty`. **⌖ Label found objects** walks an
  image's generic-find-all proposals (one at a time, highlighted, class panel
  bottom-right), then jumps to the next image that has proposals.
- **`viewAutolabel`** — five sections, top to bottom:
  - **Bootstrap** — the card for `services/bootstrap.py`: counts (labeled / empty /
    unlabeled / unreviewed / "+N since the last round"), options, the run button
    (⚡ Bootstrap → ⚡ Run round N), a hint chip, and the round's progress, built once per
    run+status and then updated in place by `pollBootstrap()` every 1.5 s (bar, message,
    log), so the log stays scrollable and Stop clickable. When a round is done it offers
    **Review round N → (count)**, and the model picker below gains the new weights.
  - **Run a model** — described below.
  - **Find all by concept — SAM 3** — *By class* (tick classes, edit each phrase) or
    *Generic — outline everything, label later* (generic nouns → proposals, then
    "Label the found objects →").
  - **Active learning** — "Rank unlabeled images" with the selected model.
  - **Bring your own weights** — `.pt` upload.

  The **Run a model** part: model picker (merges DB weights + pretrained aliases +
  RT-DETR + RF-DETR), confidence/scope controls, "replace existing" and "add new
  classes" (off by default) checkboxes.
  "Run auto-label" starts the background job and hands off to `watchAutolabelJob()`,
  which polls `/autolabel/{jid}` (rendering `job.log`/`processed`/`total` into a
  `.logbox`, same styling as training job logs) until it's done, then calls
  `renderAlPreviewGrid()`: an `.al-grid` of cards — one per labeled image, each an SVG
  preview built with the same `annotationShapesSvg()` helper Quick Review uses — with a
  per-card "Approve → To review" button (`PATCH status=review`) plus a header
  "Approve all" button that `confirm()`s before bulk-patching every not-yet-approved card
  in the batch. Nothing is auto-flagged for review anymore; the human decides per image
  or in one deliberate bulk action.
- **`viewGenerate`** — the split/preprocess/augmentation form; every augmentation
  op renders from a single `AUG_DEFS` array (name, label, `check`|`range`, min, max,
  step) so adding a new op is a one-line addition, not a new form field to hand-wire.
  "Only include approved images" defaults checked. An "Export format" select (YOLO/COCO)
  sits above the split controls — hidden entirely for classify projects, since COCO has
  no classification convention the way YOLO's folder layout does.
- **`viewVersions`** — polls every 3s while status is `building`; shows per-split
  percentages computed from `source_images`/`source_images_total` in the stored stats.
  **♥ Health** opens the version-health panel (`healthHtml()`): score, warnings, images
  per split, labels per class, objects-per-image histogram, object size/shape, duplicates.
- **`viewTrain`** — three tabs, not three views (`setTrainTab()` just toggles `hidden` on
  panes already in the DOM, all rendered up front), with one shared job list underneath
  (SSH jobs carry a 🔌 host badge, Bootstrap jobs say "bootstrap round N" instead of a
  version):
  - **Local train** — dataset-version + architecture form (pretrained YOLO aliases,
    RT-DETR aliases, and any previously trained/uploaded weight), a collapsible
    `<details class="hp-details">` box exposing every Ultralytics `train()` hyperparameter
    and built-in-augmentation kwarg from their docs (`TRAIN_HYPERPARAMS` /
    `TRAIN_AUG_DEFS` arrays, rendered by `renderHpRow()`), and the live job list with
    per-job log tailing (polled every 3.5s). Every hyperparameter row is opt-in — a
    `num`/`opt-select` row needs its own checkbox ticked before `collectTrainExtra()`
    includes it in the request; a `bool3` row is a single tri-state select
    (`(default)`/On/Off) since there's no natural "off" checkbox state distinct from
    "don't touch this." Untouched rows are simply omitted so Ultralytics' own defaults
    apply — nothing here ever silently overrides a default you didn't ask to change.
  - **SSH** — host/user/port/identity file/remote python/work dir form (saved to the
    server), **Test connection** (shows the remote torch/CUDA/GPU/ultralytics), the same
    optional hyperparameters, and "Start training on SSH host".
  - **Google Colab** — a `.train-colab` card with a solid red border (the one deliberate
    "this leaves your machine" exception to YAAP's local-only design, marked
    accordingly) containing a copyable code-snippet template
    (`buildColabSnippet()`) that reflects whichever dataset version is selected in its
    own dropdown. Explicitly a placeholder pending the user's own notebook code.

Both tabs share the `AUG_OVERLAP` map (op key → ultralytics kwarg(s), e.g.
`hflip → fliplr`) that also powers the Generate view's own snippet feature below — one
mapping, two consumers, so they can't drift apart on which augmentation corresponds to
which Ultralytics parameter.

### 4.4 `editor.js` — the `Editor` class

A single class wrapping one `<canvas>`. All annotation coordinates are stored in
**image pixel space**; a `{s, tx, ty}` view transform (scale + pan) is applied only at
draw time via `toScr()`/`toImg()` — this is what makes zoom/pan free (nothing needs
re-normalizing when you scroll the wheel).

- **Tools**: `select` (move/resize via 8 handles for bbox; for polygon: drag a vertex to
  move it, click an edge to insert a new vertex there — `_hitEdge()`/`_distToSegment()`
  find the closest point on the boundary within a 7px screen threshold — or click an
  existing vertex *without dragging* to delete it, floored at 3 vertices so a polygon
  can never collapse below a triangle), `bbox`, `polygon` (click to add vertices,
  click-near-first-vertex or double-click or Enter to close), `pan`, `sam`. The
  click-vs-drag distinction for vertex delete is a simple movement threshold: `_down()`
  records the mousedown screen position on a vertex hit, `_move()` flags `d.moved = true`
  once the pointer has traveled more than 4px, and `_up()` only deletes when that flag
  never got set — so a genuine drag, however small, always moves the vertex rather than
  removing it.
- **SAM state** lives on the instance too (`samPts`, `samBox`, `samPending`) rather than
  in `app.js`, since it has to participate in the same draw loop and hit-testing as real
  annotations — `_drawSam()` renders pending masks (dashed teal outline), the
  point/box prompt markers (teal `+` / red `−`), and the in-progress drag box.
- **Callbacks** (`opts.onChange`, `onSelect`, `onToolKey`, `onClassKey`, `onSamPrompt`,
  `onSamAccept`, `onSamState`) are how `app.js` stays in sync without the Editor knowing
  anything about the DOM outside its own canvas — a deliberate boundary: `Editor` never
  touches `document.querySelector` for anything but its own canvas element.
- **Keyboard shortcuts** are handled centrally in `_key()`: `V/B/P/H/S` switch tools,
  `1-9` and `0` (10th class) pick a class — and label a freshly drawn box that waits for one,
  `Enter` closes a polygon, accepts a pending SAM mask or labels a waiting box with the active
  class (drawing the next box does the same instead of dropping it), `Esc`
  clears SAM state or the current draft, `Delete`/`Backspace` removes the selected
  annotation, `E` (SAM tool) asks for every object like the current box
  (`onSamFindAll`). Ctrl-click toggles a pending SAM result off/on.
- **Picking and limits** — a click selects the **smallest** shape under the cursor (a selected one under
  it stays selected so it can be dragged; **Alt-click** cycles through a stack) — on dense images
  "the last one drawn" left a third of the boxes unselectable. A drag reports its change once, on
  mouseup (not per mousemove), and selecting only moves the list highlight. Boxes are clipped to the
  image on every side; a drag of more than 4 *screen* pixels makes a box (1–4 px objects at high
  zoom); inside a box smaller than 24 screen px a drag moves it instead of grabbing a handle. A window
  resize keeps zoom and pan (`_resize()`); only a new image fits. The 2nd mousedown of a double-click
  doesn't add a polygon point twice.
- **Undo / redo** — `checkpoint()` stores a JSON snapshot of `anns` just before a change (the editor's own
  gestures call it, and so does `app.js` before every edit it makes: accepting SAM results, class changes,
  deletes from the list, *Mark empty*); a drag or click that changed nothing removes its step again
  (`_dropIfUnchanged`). `undo()`/`redo()` restore a snapshot and go through `onChange`, so the result is
  autosaved; while a polygon is being drawn, `undo()` removes its last point. `load(img, anns, key)` keeps
  a history per image key (≤ 100 steps) for the life of the editor; `onHistory` updates the ↶/↷ buttons.
  Keys: `Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`, `Ctrl/Cmd+Y`; other `Ctrl` combinations go to the browser.
- **Walkthrough drawing** — `_drawSam()` also draws the "find all" walkthrough state set
  by `app.js`: the result under review (`r.cur`) with a yellow glow, labeled results in
  their class colour (`r.color`/`r.label`), the rest dimmed.

**Autosave (`viewAnnotate`)** — edits are saved 700 ms after the last change through one save at a
time; an edit made while a save is in flight is sent right after it (`editGen`), and only a save
that covered every edit shows "saved". `go()` / Exit / the rail first save and **stay** on the image
if the server didn't take it (the save keeps retrying every 5 s); closing the tab with unsaved
changes asks first and sends them with a `keepalive` request. Saves carry `expect_ids`; on a 409 the
editor merges (keeps the user's edits, adds what appeared elsewhere, accepts remote deletions only
of annotations the user didn't touch) and saves again. A SAM answer that arrives after the user
moved to another image is dropped; clicks made while SAM is busy are sent once it answers.

### 4.5 SAM-assist flow (frontend half)

`viewAnnotate` owns everything SAM-specific that isn't purely canvas-local:
1. Selecting the `sam` tool lazy-loads `/api/sam/models` once (`initSam()`) and remembers
   your last-used model variant in `localStorage` (SAM 3 is currently the only backend —
   SAM 2/MobileSAM support was removed).
2. Every click/shift-click/box-drag on the canvas fires `onSamPrompt` → `runSamPrompt()`
   → `callSam()`, which POSTs the accumulated point/box state and stores the returned
   masks on `editor.samPending`.
3. `updateSamStrip()` toggles the Accept/Discard buttons and calls
   `renderSamLabelPick()`, which renders a small floating chip row — one button per
   project class — positioned via `editor.toScr()` right above the first pending mask's
   bounding box. **Clicking a chip both labels and accepts in one action**
   (`acceptSam(clsId)`), immediately hiding itself rather than waiting on the redraw
   chain. The original Accept button (Enter key) still works too, using whichever class
   is active in the sidebar — the chip picker is additive, not a replacement.
4. `acceptSam()` converts each pending mask into a real `Annotation`: for a `detect`
   project the mask's bbox is used (segmentation → detection downgrade); for `segment`
   the full polygon is kept. SAM-made annotations are saved with `source: "model"` and
   the mask's score, even though a person placed them.
5. A `#samSimplify` range input (`0.001`–`0.03`, default `0.008` — matching the
   backend's own `SamRequest.simplify` default, which existed before the UI ever
   exposed it) sits in the SAM strip. Its `oninput` just updates a live label;
   `onchange` (on release, not every drag tick — one request per intentional change, not
   per pixel) persists the value to `localStorage` and, if a prompt is currently active
   (`editor.samPts.length || editor.samBox`), calls `runSamPrompt(editor)` again with
   the *same* accumulated points/box to re-segment at the new simplification level — no
   new click required. This is the one place `callSam()`'s request body includes
   `simplify` at all; every other SAM call path already had the field available on the
   backend and simply wasn't sending it.
6. **Warm-up.** Picking the SAM tool calls `POST /api/sam/warmup` in the background
   (~20 s once), so the first click and the first `E` don't pay the model load.
7. **"Find all like this" (`E` / the E button).** With a box drawn, `findAllLikeThis()`
   sends it as an `exemplar` prompt (min score from the strip) and gets every similar
   object in the image. `startWalk()` then walks the matches one by one
   (`focusWalk`/`revealBox` pan the canvas so the box isn't under the panel): pick a class
   (click or `1`–`9`) → next; `Del`/`X` erases a wrong match; `←`/`→` skip; `Enter` keeps
   the labeled ones (`finishWalk`); `Esc` discards. The class panel sits at a fixed spot
   bottom-right. The same walkthrough labels generic-find-all proposals
   (`startProposalWalk()`).

### 4.6 `yaap.css` — design tokens

Everything themeable is a CSS custom property on `:root` — background/panel/line/text
colors, the primary accent (`--amber`, currently yellow `#FFD23F` despite the variable
name — kept for git-diff/compat reasons across theme changes), a secondary teal for
model/success states, plus `--radius`/`--radius-pill` and the two font stacks. Swapping
the whole look is a matter of editing ~15 variables at the top of the file; nothing else
should need to change (see §7 for a caveat about this).

---

## 5. Docker packaging

`Dockerfile` has two build targets sharing one final shape:
- **`cpu`** — `python:3.11-slim` + CPU-only torch wheel.
- **`gpu`** *(default)* — `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + CUDA 12.4 torch
  wheel + `python3.11` from apt.

Both create a non-root `yaap` user (`ARG UID/GID`, default 1000/1000 — override at build
time to match your host user via `docker-compose.yml`'s `UID`/`GID` env vars) and
`chown -R yaap:yaap /app` at the end of the build. The container still starts **as
root** (no `USER` directive), because `entrypoint.sh` needs root just long enough to
`chown` the two bind-mounted volumes (`/app/data`, `/app/seg_models`) — fixing ownership
on host directories that may have been created by an earlier root-owned run — before it
execs `su -s /bin/sh yaap -c /app/start.sh` to drop privileges for everything else,
including the actual `uvicorn` process and the optional weight preload.

`docker-compose.yml` defines `yaap` (GPU, default profile) and `yaap-cpu` (opt-in via
`--profile cpu`), both binding `127.0.0.1:8811` only (not reachable from the LAN unless
you deliberately change that), both mounting `./data` and `./seg_models`.
`PRELOAD_MODELS` is a space-separated list of Ultralytics asset names downloaded once at
first boot (`start.sh` → `attempt_download_asset`, default `sam3.pt yolo11n.pt yolo26n.pt` —
the last one is what ultralytics' AMP check loads before GPU training); `start.sh` also fetches
the `Arial.ttf` font every training's dataset check wants. Pretrained aliases always resolve to
`data/models/<alias>` (`inference.alias_path()`), so preloaded files are used offline and missing
ones download there — not into the container layer. Set `PRELOAD_MODELS` to `""` to stay fully
air-gapped and upload `.pt` files through the UI instead.
Run commands inside the container as the app user — `docker compose exec -u yaap yaap …` — a root
`exec` creates root-owned files in the bind-mounted `data/` (training then falls back to a private
config folder, `trainer.worker_env()`, and says so in its log).

Additions for the newer features:
- Both images install `openssh-client`, `rsync` and `git` (SSH training; `git` because
  `requirements.txt` installs ultralytics' CLIP fork from GitHub, which SAM 3's text and
  exemplar prompts need, together with `ftfy` and `regex`).
- The GPU stage upgrades pip first: Ubuntu 22.04's apt pip (22.x) can't build
  pyproject-only packages and would install the CLIP fork as `UNKNOWN 0.0.0` with no
  `clip` module.
- `docker-compose.yml` mounts `~/.ssh` read-only at `/home/yaap/.ssh` (key auth for SSH
  training; the host user's UID must match the container's) and sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (less GPU memory fragmentation).
- `start.sh` calls `python`, not `python3`: in the GPU image `python3` is the system 3.10
  without the packages, `python` is the 3.11 that has them.

---

## 6. Data model details worth knowing

### 6.1 On-disk layout

```
data/
├─ yaap.db
├─ ssh.json            SSH training settings (no passwords)
├─ ssh_known_hosts     host keys YAAP has accepted (StrictHostKeyChecking=accept-new)
├─ .leave_at           idle shutdown armed: watch-idle.sh stops the stack 60 s after this time
├─ .leave_pending      tab closed while a job ran: becomes .leave_at when the job ends
├─ models/            global + uploaded .pt cache (also YOLO_CONFIG_DIR)
├─ runs/               job_<id>.log + raw ultralytics run folders
│                      (job_<id>/dataset/ = Bootstrap's temporary training set, deleted after training)
└─ projects/<pid>/
   ├─ images/          originals, exif-transposed once at upload
   ├─ thumbs/          320px JPEG previews
   └─ versions/v<id>/  generated dataset (train/val/test + data.yaml) + v<id>.zip
seg_models/             SAM .pt weights (sam3.pt — the only SAM backend now)
```

### 6.2 Annotation `data` shapes

| `kind` | `data` |
|---|---|
| `bbox` | `{"x": float, "y": float, "w": float, "h": float}` — top-left + size, image pixels |
| `polygon` | `{"points": [[x, y], ...]}` — image pixels, in order |
| `classification` | `{}` — the class_id itself is the label; at most one per image |

### 6.3 `ImageAsset.status` state machine

Seven values. The backend stores whatever string `PATCH /images/{iid}` sends (no enum),
so the frontend is what keeps these consistent.

- `unannotated` → default; set back to this whenever the annotation list becomes empty
  (editor save, the Review grid's 🗑 erase, or clearing an `empty` flag).
- `annotated` → set whenever an annotation save has at least one shape (editor
  autosave, import/merge). **Every save recomputes status from the annotation count**,
  so editing an `approved`, `review` or `needs_reannotation` image drops it back here.
  This is deliberate: approval is a snapshot, and fixing a rejected image is what sends
  it back to the queue.
- `auto-annotated` → set by an auto-label job, SAM "find all" and a Bootstrap round on
  every image they added predictions to. Nothing a model produced counts as reviewed
  until a human looks at it. These jobs only write an image whose status is unchanged
  since they picked it, so a label, approval or rejection made meanwhile always wins.
- `review` → "graduated from auto-label into the main QC queue": set only by the
  auto-label preview grid's per-image **Approve** and its **Approve all** button.
- `needs_reannotation` → "a human rejected this, fix it": set by **Reject** (editor
  topstrip, the Review grid's ✕, Quick Review's Reject / `R` key, and the approved-images
  browser's "Revert"). Reject toggles: clicking it again clears the flag back to
  `annotated`. It used to share the `review` string with auto-label approval, which made
  rejected images show up under both "To review" and "Needs re-annotation" at once; it
  is a separate value now so the two never overlap.
- `empty` → "checked, there are no objects here": set by the editor's **∅ Mark empty**,
  which also clears the image's annotations. Counts as labeled in the project's
  `annotated_count` (as do `annotated` and `approved`; `auto-annotated` and `review` are
  counted as `predicted_count`, `needs_reannotation` as neither). Bootstrap trains on
  exactly these three statuses.
- `approved` → set only by an explicit Approve (editor topstrip, the Review grid's ✓,
  Quick Review's Accept / `A` key). Version generation defaults to `approved_only`.

The Review view's filter chips (`QC_FILTERS` / `qcMatch()` in `app.js`, §4.3) read this
as: **To review** = `annotated ∪ auto-annotated ∪ empty ∪ review` (still needs a
decision — a pseudo-filter, not a status), **Needs re-annotation** =
`needs_reannotation`, and **Unannotated** / **Auto-annotated** / **Approved** = exact
matches. `needs_reannotation` is deliberately *not* in "To review": a rejected image
stays out of the queue until someone fixes and saves it.

### 6.4 Class matching convention

Three independent code paths — autolabel (`models.py`), zip import, and project merge
(both in `importer.py`) — all resolve model/foreign class names against the target
project **by exact name string**, creating a new `LabelClass` (next `PALETTE` color, next
`order_idx`) only when no existing class matches. This is what lets you e.g. auto-label
with one model, then import a Roboflow export, then merge in another project, and have
them all converge onto the same class list instead of duplicating it three ways —
as long as the names match exactly (case-sensitive, no fuzzy matching).

---

## 7. Known quirks / gotchas (learned the hard way this session)

- **`[hidden]` vs `display:` in the same CSS file.** Several components
  (`.btn`, `.rail`, `.sam-strip`, `.sam-label-pick`, …) declare an explicit `display:`
  value. Per the CSS cascade, an author stylesheet's `display` **always beats** the
  browser's built-in `[hidden]{display:none}` rule at equal specificity — meaning
  `element.hidden = true` silently does nothing on any element that also matches one of
  those classes. Fixed once, globally, with `[hidden]{display:none!important}` near the
  top of `yaap.css` — if you add a new component with its own unconditional `display:`
  rule and also toggle it via `.hidden`, this is already covered; don't remove that rule.
- **Static file caching.** Plain `StaticFiles` sends no `Cache-Control` header, so
  browsers heuristic-cache `yaap.css`/`app.js`/`editor.js` across edits. `app.py` now
  mounts `NoCacheStatic` (`Cache-Control: no-cache`, cheap 304s); if a change still
  doesn't show up, hard-refresh (Ctrl/Cmd+Shift+R).
- **Text prompts are back, but only for "find all".** The editor's click/box prompts are
  still `point`/`box` only. Text phrases and box exemplars go through ultralytics'
  `SAM3SemanticPredictor`, used by the `exemplar` prompt type and the find-all job —
  and need ultralytics' CLIP fork installed (§5); without it the error message says how
  to install it.
- **Declare `let` state before the code that reads it.** `viewAnnotate` once declared
  `let walk` after `new Editor(...)`, whose first draw already read it — a temporal-dead-
  zone `ReferenceError` that left every image blank in every project. Keep state
  declarations at the top of a view.
- **Training data must be copies.** ultralytics re-saves a JPEG it considers corrupt
  (bytes after the end marker — common on phone photos) *in place* while checking a
  dataset. Hard links would let it rewrite the user's originals; Bootstrap copies.
- **Background jobs vs. a person.** A job that read an image's state and writes it later
  must re-check under the write lock (`jobs.claim_image`) — inference takes long enough
  for the user to save, approve or reject the same image in between.
- **A CUDA error that never clears** ("CUDA device busy", "expected str … NoneType" from
  SAM) means the server process' CUDA context died; restart the container.
- **`ultralytics` needs `timm` for SAM 3.** Without it pinned in `requirements.txt`,
  Ultralytics auto-installs `timm` into the running container at request time — which
  then requires a restart to actually take effect (the running Python process already
  cached the failed import). It's pinned now specifically to avoid this on fresh builds.
- **Root-owned bind mounts.** Before the non-root user was added, any file the container
  created under `./data` ended up owned by `root` on the host, which then blocked the
  host user from touching it directly (and caused a `sqlite3.OperationalError: attempt to
  write a readonly database` if the DB file itself lost write access). `entrypoint.sh`'s
  one-time `chown` on container start is what heals this on old volumes.

---

## 8. Where to extend things

- **New augmentation op** → add one entry to `AUG_DEFS` in `app.js` (name, label, control
  type, range) and handle the same key in `augmenter.build_augment()`. No other file
  needs to change.
- **New task type** (beyond detect/segment/classify) → touches `ProjectCreate.task_type`
  pattern, `exporter.generate()`'s per-task branches, `_write_label_txt()`,
  `inference.predict()`'s per-task branches, and `viewAnnotate`'s tool visibility logic.
  It's the single most invasive change in the codebase — every export/import/inference
  path branches on `task_type` explicitly rather than through a task-type abstraction.
- **New model backend for auto-label** → if it shares ultralytics' `Results` shape (like
  `RTDETR` does), just add a loader function and a family-detection check (follow
  `_load_rtdetr`/`is_rtdetr_arch`) — `predict()`'s task branching needs no changes. If it
  has its own separate API (like `rfdetr`), follow the `_load_rfdetr`/`_predict_rfdetr`
  pattern instead. Either way, surface it in `models.py`'s `/api/models` response and add
  it to the `viewAutolabel`/`viewTrain` dropdowns.
- **New export format** → follow the `fmt == "coco"` branch in `exporter.generate()`: it
  only needs to change how labels get written per image (and whether/what manifest file
  gets written after the loop) — the preprocessing/augmentation pipeline before it is
  format-agnostic and needs no changes.
- **New SAM backend** → add an entry to `SAM_MODELS` in `services/sam.py`; the frontend's
  `#samModel` dropdown picks up whatever `/api/sam/models` returns automatically.
- **New database column** → just add the `Column` to the model in `db.py`; `init_db()`
  adds it to existing databases on the next start. Give it a plain `default=` so existing
  rows get a value. Anything else (rename, data fix, unique/primary-key column) → append
  a function to `MIGRATIONS` in `db.py` and test it on an old database
  (`tests/test_migrations.py` shows how).
- **New background job** → if it writes model output to images, snapshot each image's
  status when the job starts and write through `jobs.claim_image()`; call
  `jobs.job_finished()` in its `finally`, so a shutdown deferred while it ran can go
  ahead; and remember that in-process jobs die with the server
  (`fail_interrupted_jobs()` marks them failed at the next start).
- **New table that points at a project** → also delete its rows (and any files) in
  `cleanup.delete_project()`.
