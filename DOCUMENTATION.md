# YAAP — Architecture & Code Documentation

YAAP (Yet Another Annotation Platform) is a local-first, self-hosted image annotation
tool developed by ComputerVisionAIHub, and deliberately small: one FastAPI
process, one SQLite file, static HTML/CSS/vanilla-JS frontend (no build step, no
framework), and Ultralytics for every model-backed feature (auto-label, SAM-assisted
segmentation, training). Everything — images, labels, model weights, training runs —
lives under `./data` on the host, bind-mounted into the container. Nothing is sent
anywhere over the network unless you explicitly download a pretrained weight.

This document explains how the pieces fit together: the backend's routers/services/db
layers, the frontend's view-router and canvas editor, the Docker packaging, and the
on-disk data layout, plus the end-to-end flows that tie them together (annotate → QC →
generate → train, and SAM-assisted segmentation).

---

## 1. High-level architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│   Browser (vanilla JS)      │  HTTP  │           FastAPI (uvicorn)           │
│                              │◄──────►│                                        │
│  index.html                 │  JSON  │  app.py — mounts routers + /static     │
│  ├─ api.js   (fetch client) │        │                                        │
│  ├─ editor.js (canvas)      │        │  routers/  — one file per resource     │
│  └─ app.js   (hash router,  │        │  services/ — business logic, no HTTP   │
│      all views)              │        │  db.py     — SQLAlchemy models         │
│  yaap.css (design tokens)   │        │  schemas.py — Pydantic I/O contracts   │
└─────────────────────────────┘        └──────────────┬─────────────────────────┘
                                                        │
                                          ┌─────────────┴─────────────┐
                                          │     ./data  (bind mount)   │
                                          │  yaap.db          (SQLite) │
                                          │  projects/<id>/…   images  │
                                          │  models/…      weight cache│
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
the entire dev loop. The backend has no ORM migrations — `Base.metadata.create_all()`
runs on every startup, so schema changes require either a fresh `data/yaap.db` or a
manual `ALTER TABLE`.

---

## 2. Directory layout

```
YAAP/
├─ backend/
│  ├─ app.py              FastAPI app: router registration + static mount + "/"
│  ├─ config.py           All paths (DATA_DIR, PROJECTS_DIR, MODELS_DIR, SEG_MODELS_DIR…)
│  ├─ db.py                SQLAlchemy engine/session + all table models
│  ├─ schemas.py           Pydantic request bodies (validation only, no ORM coupling)
│  ├─ routers/             One FastAPI router per resource — thin HTTP glue
│  │  ├─ projects.py       projects + classes CRUD
│  │  ├─ images.py         image upload/list/file/thumb/patch/delete
│  │  ├─ annotations.py    per-image annotation get/replace (editor autosave)
│  │  ├─ models.py         pretrained/weights listing, .pt upload, autolabel
│  │  ├─ sam.py            SAM prompt endpoint + model listing
│  │  ├─ datasets.py       augment preview, raw export, import, merge, versions
│  │  └─ training.py       training job start/list/log/stop
│  └─ services/            Business logic — no FastAPI/HTTP imports, all pure Python
│     ├─ storage.py        Project folder layout, upload dedup, thumbnailing
│     ├─ sam.py            SAM model loading + point/box prediction → polygons
│     ├─ inference.py      YOLO/RF-DETR loading + prediction → YAAP annotation dicts
│     ├─ augmenter.py      albumentations pipeline builders (preprocess + augment)
│     ├─ exporter.py       Dataset version generation (split/preprocess/augment/zip)
│                          + export_raw() — no-version, no-split zip pack
│     ├─ importer.py       Zip → project (import_zip) and project → project (merge_project)
│     ├─ trainer.py        Launches/monitors/stops the training subprocess
│     └─ train_worker.py   Standalone script run as that subprocess (actual model.train())
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
├─ data/                    Bind-mounted; DB + per-project images/thumbs/versions
├─ seg_models/               Bind-mounted; dedicated SAM weights directory
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

Creates the `FastAPI()` instance, calls `init_db()` (creates tables if missing), registers
every router, mounts `frontend/` at `/static`, and serves `frontend/index.html` at `/`.
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

### 3.3 `db.py` — the whole schema, seven tables

SQLite via SQLAlchemy 2.0, one global `engine`/`SessionLocal`. `get_db()` is the FastAPI
dependency every router uses (`Depends(get_db)`), a generator yielding a `Session` and
closing it after the request.

| Table | Purpose | Notable columns |
|---|---|---|
| `Project` | one dataset/task | `task_type`: `detect`\|`segment`\|`classify` — fixed at creation, drives every downstream tool choice |
| `LabelClass` | a class within a project | `order_idx` (becomes the YOLO class index on export — **keep stable once training starts**), `color` |
| `ImageAsset` | one uploaded image | `status`: `unannotated`\|`annotated`\|`review`\|`approved` (see §6.3); `split`: `""`\|`train`\|`val`\|`test` — a **manual pin** that `_split_images()` respects before randomizing the rest |
| `Annotation` | one shape on one image | `kind`: `bbox`\|`polygon`\|`classification`; `data` is a JSON string (`{x,y,w,h}` for bbox, `{points:[[x,y]…]}` for polygon, `{}` for classification); `source`: `manual`\|`model`; `confidence` |
| `DatasetVersion` | a frozen, generated snapshot | `config` (the full `VersionCreate` body, JSON), `stats` (output of `exporter.generate()`), `status`: `building`\|`ready`\|`failed` |
| `ModelWeight` | an uploaded or trained `.pt` | `project_id` nullable = global weight available to all projects; `source`: `uploaded`\|`trained` |
| `TrainJob` | one training run | `pid` (OS process id, used to `stop()`/`refresh_status()`), `status`: `queued`\|`running`\|`done`\|`failed`\|`stopped`, `weights_path` (best.pt once done) |

`Annotation.data_obj` is a convenience property that lazily `json.loads`es `data`.

### 3.4 `schemas.py` — request bodies only

Pure Pydantic models used for **input validation**, never returned directly (every router
hand-rolls its own `serialize_*()` dict instead — see below). Notable ones:
- `VersionCreate.approved_only` defaults to `True` — Generate refuses to run unless
  images are QC-approved (or you explicitly uncheck the box in the UI).
- `MergeRequest` — just `source_project_id` + optional `default_split`.
- `TrainRequest.extra` — an escape hatch dict merged straight into `model.train(**extra)`
  in `train_worker.py`, for any Ultralytics kwarg not exposed in the UI.

### 3.5 Routers — thin HTTP layer

Every router follows the same shape: parse path params, load the row (`project_or_404` /
`_img` helpers), delegate to a service function, catch the service's domain exception
(`RuntimeError`, `ValueError`, `SamError`, `InferenceError`) and turn it into an
`HTTPException` with a human-readable message, return a plain dict (FastAPI JSON-encodes
it). None of the routers talk to the ORM beyond simple lookups — all real logic lives in
`services/`.

- **`projects.py`** — also defines `PALETTE` (10 hex colors auto-assigned to new classes)
  and the shared `project_or_404()` helper imported by almost every other router.
  `serialize_project()` computes `annotated_count` on the fly (`status=="annotated" or
  has annotations`) rather than storing a counter.
- **`images.py`** — upload accepts a `list[UploadFile]`; unsupported extensions go to
  `skipped` rather than erroring the whole batch. `_img()` is the per-image
  ownership-check helper reused by `annotations.py` and `sam.py`.
- **`annotations.py`** — a single `PUT` that **replaces the entire annotation set** for
  an image (delete-all-then-insert). This is what the editor's autosave calls every time
  you stop drawing for 700ms (see `markDirty()`/`save()` in `app.js`). Also flips
  `image.status` between `annotated`/`unannotated` based on whether the new set is empty.
- **`models.py`** — `/api/models` lists three sources merged into one dropdown: DB-stored
  `ModelWeight` rows (project-specific + global), `PRETRAINED_MODELS` aliases (Ultralytics
  downloads these on first use), and RF-DETR variants if the `rfdetr` package is
  installed. `/autolabel` is the heaviest endpoint: for each image, calls
  `inference.predict()`, optionally auto-creates missing classes (matched **by name**,
  same pattern reused in `importer.py`), and marks touched images `status="review"` so a
  human has to pass over model output before it counts as ground truth.
- **`sam.py`** — two endpoints: `GET /api/sam/models` (static capability list) and
  `POST /projects/{pid}/images/{iid}/sam` (one point/box prompt → candidate masks).
  Stateless on the backend — the frontend accumulates click history and resends the full
  point/box set on every prompt.
- **`datasets.py`** — the biggest router: augmentation preview (single contact-sheet
  JPEG, no zip), the "just pack what I have" raw export, dataset import, project-to-project
  merge, and the versioned Generate pipeline (`POST /versions` kicks off
  `_build_version()` as a FastAPI `BackgroundTasks` job — the HTTP response returns
  immediately with `status="building"`, and the frontend polls `GET /versions` every 3s
  until it flips to `ready`/`failed`).
- **`training.py`** — starts `trainer.start()`, which spawns a **separate OS process**
  (not a background task) so a crash or `SIGTERM` can't take the web server down with it,
  and so `stop()` can kill it by PID group.

### 3.6 Services — the actual logic

- **`storage.py`** — defines the on-disk layout (`data/projects/<pid>/{images,thumbs,versions}`)
  and the one function everything else calls to persist an upload: `save_upload()`.
  It de-duplicates filenames by appending `_1`, `_2`, … , `exif_transpose()`s the image
  once at ingest (so rotation metadata never has to be dealt with again downstream), and
  builds a 320px JPEG thumbnail. Both `importer.py` and the plain upload endpoint funnel
  through this single function, so dedup/thumbnailing behavior is consistent everywhere.

- **`sam.py`** *(service)* — lazy-loads an Ultralytics `SAM` model
  (`functools.lru_cache(maxsize=2)`, so switching between two SAM variants doesn't
  reload from disk every request), checks `SEG_MODELS_DIR` before `MODELS_DIR` before
  falling back to Ultralytics' own alias/download resolution. `predict()` only accepts
  `point`/`box` prompts (text-prompt support was deliberately removed — see §7). Masks
  come back from Ultralytics as raw arrays; `mask_to_polygons()` runs
  `cv2.findContours` + `approxPolyDP` to turn each into a simplified pixel-space polygon,
  filtering out anything under `min_area=60`.

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
  nothing for existing YOLO weights.

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
    `config["format"] == "coco"` additionally writes, per split, one flat copy of the
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
    images per split, used by the Versions view to show real percentages), and `format`.
  - `export_raw(project, approved_only)` — the "just give me a zip" path: no
    `DatasetVersion` row, no split, no preprocessing, no augmentation, **YOLO format
    only** (no format choice here — this is the quick-pack path, not the versioned one).
    Copies images as-is into `images/`+`labels/` (or class folders), writes `classes.txt`
    + a minimal `data.yaml`, zips, and deletes its own scratch folder — rebuilt fresh on
    every request so it always reflects current annotations.

- **`importer.py`** — the inverse of `exporter.py`, plus project-to-project merge:
  - `import_zip()` sniffs a `data.yaml`/`classes.txt` for class names, then treats **any**
    image file in the zip as an entry (matched to a same-stem `.txt` label file by
    filename only, ignoring folder structure — so both `images/x.jpg`+`labels/x.txt` and
    nested `train/images/x.jpg`+`train/labels/x.txt` layouts work identically), pins
    `image.split` from any `train`/`val`/`test` path component it finds. Classes are
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
  DB still says `running` but the PID is dead, it flips the job to `failed` without
  ever having been told explicitly (covers the container-restarted-mid-training case).

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
which keeps `app.js` free of any `fetch()` calls at all.

### 4.3 `app.js` — router + every view

A tiny hash router: `#/` → `viewHome()` (project grid), `#/p/<id>/<view>` → one of the
view functions in the big dispatch table in `route()`. Global mutable state lives in one
object, `S` (current project, image list, current image index, the live `Editor`
instance, poll-interval handles). Each view function is self-contained: it sets
`$("#stage").innerHTML`, wires up event handlers, and does its own data fetching — there's
no shared component model, view functions just fully re-render on every navigation.

Views, in rail order:

- **`viewUpload`** — dropzone (drag or click-to-browse), the image grid, and the
  "Dataset in / out" card: raw zip download (`API.exportRawUrl` + a synthetic `<a
  href>.click()`), zip import, and the merge-from-another-project dropdown (only lists
  projects with a matching `task_type`).
- **`viewReview`** *(the QC screen)* — filter chips with live counts; each grid tile has
  an "open in editor," a quick-approve, and a quick-reject button (reject sets
  `status="review"`, reusing the same status autolabel uses for "needs a human pass" —
  click again to clear it). Clicking the tile itself also opens the editor
  (`openReviewImage` reassigns `S.images`/`S.curImage` and navigates to `annotate`, so
  Prev/Next inside the editor walks the same filtered ordering you were browsing). The
  default filter is a pseudo-status, `"reviewable"` (labeled "To review" in the UI) —
  `qcMatch()` defines it as `status === "annotated" || status === "review"`, i.e.
  *deliberately excluding* both `unannotated` (nothing to review yet) and `approved`
  (already resolved). This is what makes the queue actually drain to empty as you
  approve/reject your way through it, rather than staying permanently populated — an
  earlier version defined it as "anything not unannotated," which included `approved`
  and meant reviewed images never left the list.
- **`viewClasses`** — name (contenteditable, saved on blur) + color picker + delete per
  row; the row order **is** the export class index, called out explicitly in the copy.
- **`viewAnnotate`** — the canvas editor screen; see §4.4/§4.5, this is the largest and
  most stateful view.
- **`viewAutolabel`** — model picker (merges DB weights + pretrained aliases + RT-DETR +
  RF-DETR), confidence/scope controls, and a "bring your own weights" `.pt` uploader.
- **`viewGenerate`** — the split/preprocess/augmentation form; every augmentation
  op renders from a single `AUG_DEFS` array (name, label, `check`|`range`, min, max,
  step) so adding a new op is a one-line addition, not a new form field to hand-wire.
  "Only include approved images" defaults checked. An "Export format" select (YOLO/COCO)
  sits above the split controls — hidden entirely for classify projects, since COCO has
  no classification convention the way YOLO's folder layout does.
- **`viewVersions`** — polls every 3s while status is `building`; shows per-split
  percentages computed from `source_images`/`source_images_total` in the stored stats.
- **`viewTrain`** — two tabs, not two views (`setTrainTab()` just toggles `hidden` on two
  panes already in the DOM, both rendered up front):
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
    Carries a visible "known issue" banner since local training doesn't currently
    complete successfully; not yet root-caused for lack of a reproducible error/log.
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
  `1-9` picks a class, `Enter` closes a polygon or accepts a pending SAM mask, `Esc`
  clears SAM state or the current draft, `Delete`/`Backspace` removes the selected
  annotation.

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
   the full polygon is kept.
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
first boot (`start.sh` → `attempt_download_asset`); set it to `""` to stay fully
air-gapped and upload `.pt` files through the UI instead.

---

## 6. Data model details worth knowing

### 6.1 On-disk layout

```
data/
├─ yaap.db
├─ models/            global + uploaded .pt cache (also YOLO_CONFIG_DIR)
├─ runs/               job_<id>.log + raw ultralytics run folders
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

- `unannotated` → default; set back to this if the annotation list becomes empty.
- `annotated` → set whenever the editor saves a non-empty annotation list, or an import/
  merge brings one in with a classification tag.
- `review` → set two ways: by **autolabel**, to force a human pass over model output
  before it's trusted, and by the manual **Reject** action (editor topstrip button, or
  the Review grid's quick-reject) — both share the same status rather than needing a
  separate "rejected" state, so a rejected image lands right back in the "To review"
  queue once it's fixed. Reject toggles: clicking it again on an already-`review` image
  clears the flag back to `annotated`.
- `approved` → set only by the explicit Approve action (editor topstrip button, or the
  Review grid's quick-approve). **Editing an approved image's annotations silently drops
  it back to `annotated`** (the save handler always recomputes status from the current
  annotation count) — approval is a snapshot judgment, not a sticky flag, by design: it
  forces re-approval after any change. Version generation defaults to filtering on this
  status.

The Review view's default filter (§4.3, `viewReview`) reads this state machine as: **To
review** = `annotated ∪ review` (still needs a decision), **All** = everything,
**Approved**/**Needs review**/**Unannotated**/**Annotated** = exact single-status
matches. Nothing new on the backend — `PATCH /images/{iid}` already accepted an
arbitrary `status` string; reject is purely a frontend convention reusing it.

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
- **Static file caching.** `StaticFiles` sends no `Cache-Control` header, so browsers can
  aggressively heuristic-cache `yaap.css`/`app.js`/`editor.js` across edits. If a change
  doesn't seem to show up, hard-refresh (Ctrl/Cmd+Shift+R) before assuming the server is
  stale.
- **Text-prompt SAM was removed on purpose** (not partially — both the ultralytics
  concept-prompt path and the `facebookresearch/sam3` native-package fallback were
  deleted from `services/sam.py`, and the `type` pattern in `routers/sam.py` no longer
  accepts `"text"`). Only `point`/`box` prompts exist now.
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
