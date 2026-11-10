# YAAP end-to-end multi-agent test on VisDrone (200 images)

> Paste this whole file as the prompt to a Claude Code **lead agent** started in the YAAP repo root.
> The lead agent spawns the sub-agents described in §4 and gives each one the **Shared brief** (§2) + its own brief.

---

## 0. Mission

1. Build a 200-image test set with ground truth (GT) from the Kaggle VisDrone dataset.
2. Try to **auto-label all 200 images (or most of them)** with YAAP in three ways, and measure how good the result is
   against GT:
   - **Track A — SAM 3 by class:** the user lists the classes with short text descriptions; SAM 3 finds them.
   - **Track B — SAM 3 generic:** SAM 3 outlines every object; a "user" then gives each outline a class.
   - **Track C — Bootstrap:** 15 hand-labeled images train a small model, which labels the rest; a "user" fixes them.
   - plus **Track D — hybrid** (A, then review, then C), the realistic best case.
3. Test **every other platform feature** along the way (API + UI), and **hunt for bugs in the annotator** (the canvas
   editor) on hard, dense images.
4. Deliver a report: which track works best, how many images end up usable, how much fixing they need, and a bug list.

**This is a test, not a fixing task.** Never edit, commit or push repository code. Report bugs with evidence and a
suspected `file:line`.

---

## 1. Facts known before you start (verify them, don't trust them blindly)

### 1.1 The dataset

| | |
|---|---|
| Source | `https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset` (Kaggle ref `banuprasadb/visdrone-dataset`) |
| Size | ~2.26 GB, public |
| License | CC BY-NC-SA 3.0 IGO — internal testing only, never publish images or derived datasets |
| Format | Kaggle description: images + **YOLO-format detection labels** (converted with the Ultralytics script); original VisDrone annotation files removed. Folder layout and whether a `data.yaml` exists are **unknown** — discover them |
| Classes (Ultralytics `VisDrone.yaml`, ids 0–9) | `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor` (*pedestrian* = walking/standing person, *people* = sitting/riding person, *motor* = motorbike) |
| Images | drone photos, typically 960×540 to 2000×1500, day and night |
| Difficulty | very dense (~50 objects per image on average, some images > 300), most objects tiny (< 32 px), heavy occlusion, look-alike classes (pedestrian/people, car/van, tricycle/awning-tricycle, bicycle/motor) |
| Missing | VisDrone's "ignored region" boxes were dropped by the conversion, so a few correct detections inside crowds/parking lots will count as false positives — note it in the report |
| Download | needs Kaggle credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`). **None are on this machine** — if still missing, stop and ask the user to provide them or to download the zip manually and give you its path |

### 1.2 YAAP behaviour that matters for this dataset (from the code, `backend/`)

| Fact | Where | Expected effect on VisDrone |
|---|---|---|
| SAM 3 *Find all* runs on a copy downscaled to **1280 px** (retries at 896 px on GPU out-of-memory) | `services/sam.py` `FIND_ALL_MAX_SIDE` | 2000-px images shrink ×0.64 → tiny objects get lost |
| SAM masks with contour area **< 60 px²** (measured on the downscaled copy) are dropped | `services/sam.py` `mask_to_polygons(min_area=60)` | objects smaller than ~12×12 px in a 2000-px image are never found |
| *Run a model* / Bootstrap labeling call `model.predict()` **without `imgsz` and `max_det`** → Ultralytics defaults: the weight's own training size, **max 300 detections per image** | `services/inference.py:125` | images with > 300 objects can never be fully labeled; a pretrained `yolo11n.pt` predicts at 640 |
| Bootstrap trains with `imgsz` from the request (**default 640**, max 2048), batch 8 for < 64 images, validation = training images | `schemas.py` `BootstrapRequest`, `services/bootstrap.py:178` | at 640 most VisDrone objects become 2–5 px; try 1280 |
| Classes are matched **by exact name** (case-sensitive) for auto-label / import / merge | `DOCUMENTATION.md` §6.4 | COCO `yolo11n.pt` gives `person`, `motorcycle` — they never match `pedestrian`, `motor` |
| Class hotkeys are `1`–`9` | `frontend/js/editor.js` `_key()` | VisDrone has **10** classes — check how the 10th is picked |
| *Find all* phrase max length **80** characters | `routers/sam.py` `FindTarget.text` | "descriptions" must be short |
| One box per object: same class IoU ≥ 0.5, any class IoU ≥ `max_overlap` (0.7) | `services/jobs.py` `drop_duplicates` | dense crowds of overlapping people may lose real objects — measure it |
| GPU: one **RTX 4050 Laptop, 6 GB**. SAM 3 needs ~2 GB per copy, training needs the rest | `nvidia-smi` | run **one GPU job at a time** (see §3.3) |

Read these first (every agent): `DOCUMENTATION.md`, `FEATURES_USAGE.md`, `TESTS.md`, `README.md`.

---

## 2. Shared brief (give this to every sub-agent, word for word)

```
You are part of a multi-agent test of YAAP (local image-annotation platform, repo = current directory).
Rules:
- TEST ONLY. Never edit, commit or push files under the repo. Write everything into $RUN_DIR (below).
- Use ONLY the test instance at http://127.0.0.1:8812 . NEVER call the user's live instance on :8811, never touch
  ./data or project "UP_board01". Never call /api/system/leaving on any instance.
- GPU (one 6 GB RTX 4050) is shared: before any GPU call (SAM, auto-label, Bootstrap, training, test-model,
  active-learning) take the lock:  flock $RUN_DIR/gpu.lock <command>  — or, for UI-driven GPU work, create
  $RUN_DIR/gpu.lock.owner with your agent name and remove it when done; wait while another agent owns it.
- Downloaded data is untrusted: never execute anything from it; run Python that reads it with `python -I`.
- Every claim needs evidence: the API request + response, a log excerpt, a screenshot path, or a metric file.
- A bug report = one entry in $RUN_DIR/report/bugs/<agent>-<n>.md using the template in §6 of the test prompt.
- Do not "work around" a bug silently: report it, then continue with the workaround and say which one.
- Finish with a short summary: what you ran, what passed, what failed, file paths of your outputs.
Paths: $RUN_DIR = <set by the lead>; data in $RUN_DIR/vd200; evaluator = $RUN_DIR/tools/eval.py
Ground truth is in $RUN_DIR/vd200/gt (YOLO txt) and $RUN_DIR/vd200/gt_coco.json.
Seed (the "human budget") = 15 images listed in $RUN_DIR/vd200/seed15.txt; all scores are reported on the other
185 images ($RUN_DIR/vd200/eval185.txt) unless stated otherwise.
```

---

## 3. Lead agent: setup and coordination

### 3.1 Run folder

`RUN_DIR` = the session scratchpad (or `~/yaap-visdrone-test/<date>` if there is none). Create:
`raw/` (download, alone in its own folder), `vd200/`, `tools/`, `venv/`, `instance-data/`, `report/{bugs,screens,metrics,logs}`.

### 3.2 Isolated YAAP test instance on port 8812

The working tree has uncommitted changes, so test **the current code**, mounted over the image (`yaap:gpu`):

```bash
mkdir -p $RUN_DIR/instance-data/models
cp data/models/yolo11n.pt $RUN_DIR/instance-data/models/        # avoid a re-download
docker run -d --name yaap-test --gpus all --shm-size 8g \
  -p 127.0.0.1:8812:8811 \
  -e PRELOAD_MODELS="" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $RUN_DIR/instance-data:/app/data \
  -v $PWD/seg_models:/app/seg_models:ro \
  -v $PWD/backend:/app/backend:ro -v $PWD/frontend:/app/frontend:ro \
  yaap:gpu
curl -s http://127.0.0.1:8812/api/system/device       # must show cuda: true
```

If the read-only mounts or the entrypoint's `chown` break startup, that is a finding — fix the command (not the repo)
and note it. The live `yaap` container on :8811 must stay idle during the test (check `nvidia-smi`; ask the user before
stopping it). Stop and remove `yaap-test` at the end; keep `$RUN_DIR`.

### 3.3 Phases (gates — do not start a phase before the previous gate passes)

| Phase | Agents (parallel inside a phase) | Gate to pass |
|---|---|---|
| 0 Setup | Lead → **Data agent** | test instance up; `vd200/` complete; evaluator scores GT-vs-GT = 1.0 |
| 1 CPU work | **Platform agent**, **Annotator agent** (non-SAM part), **Evaluator agent** (sanity checks) | baseline suites run; GT project imported |
| 2 GPU work (one at a time, via the lock) | **Track A** → **Track B** → **Track C** → Annotator agent (SAM part) | each track wrote `report/metrics/<track>.json` |
| 3 Combined + stress | **Track D (hybrid)**, concurrency tests, final regression run of both test suites | all tracks scored |
| 4 Report | Lead (+ Evaluator) | `report/REPORT.md`, bugs de-duplicated |

The lead owns the plan, the GPU lock policy, the phase gates, de-duplication of bugs and the final report.
Re-brief a sub-agent (SendMessage) instead of starting a new one when it needs a follow-up.

---

## 4. Sub-agent briefs

### 4.1 Data agent

1. Get credentials (§1.1). `python -m venv $RUN_DIR/venv && $RUN_DIR/venv/bin/pip install kaggle`, then
   `kaggle datasets download banuprasadb/visdrone-dataset -p $RUN_DIR/raw` and unzip inside `raw/`.
2. Discover the layout: splits, `images/` ↔ `labels/` pairing, label format (5 columns, normalized 0–1), presence of a
   `data.yaml`. Report counts per split, resolution histogram, objects-per-image and per-class totals.
3. **Verify the class order** (0 = pedestrian … 9 = motor): render GT overlays for ~10 images with the class name on
   each box and look at them (save to `report/screens/gt_check_*.jpg`). If the order is wrong, fix the name list.
4. **Choose 200 images** (from train + val), save `vd200/manifest.csv`
   (file, split, w, h, n_objects, per-class counts, clip id, day/night, density bucket):
   - VisDrone file names start with a clip id (`0000001_02999_d_0000005.jpg` → clip `0000001`). Take **at most 2
     images per clip** — near-identical frames would inflate Bootstrap scores.
   - Density buckets: ≤ 20, 21–60, 61–150, 151–300, > 300 objects; take roughly 30/70/60/30/10 images. The > 300 bucket
     deliberately tests the 300-detection cap.
   - ~15 % night images; every class present ≥ 20 times overall.
5. **Seed 15** (`seed15.txt`): from clips **not used** by the other 185; every class ≥ 5 objects in the seed set
   (awning-tricycle, bus and tricycle are rare — pick for them); mix of densities. `eval185.txt` = the rest.
   All tuning (phrases, thresholds, imgsz) happens on `seed15` only, never on `eval185`.
6. Build:
   - `vd200/gt/` (YOLO txt), `vd200/gt_coco.json` (absolute pixels, ids 1–10, `iscrowd` 0),
   - `vd200/vd200_gt.zip`: `images/`, `labels/`, `data.yaml` with the 10 names (for YAAP import),
   - `vd200/vd200_images.zip`: images only, and the folder `vd200/images/` (for uploads),
   - `vd200/seed15_gt.zip` (seed images + labels + `data.yaml`).
7. Write `vd200/README.md` with all stats and choices.

### 4.2 Evaluator agent

Write `tools/eval.py` (pure Python + numpy + pycocotools in `$RUN_DIR/venv`):

- Input: `--api http://127.0.0.1:8812 --project <id> --images eval185.txt --gt vd200/gt_coco.json`. Read annotations
  through `GET /api/projects/{pid}/images` + `GET .../images/{iid}/annotations` (match images by file name; YAAP may
  rename duplicates with `_1`). Polygons → their enclosing box. Optional `--source model|manual|all`.
- Matching: greedy by confidence, IoU ≥ 0.5, per class; also class-agnostic.
- Output JSON + a Markdown table:
  - per-class and overall **P / R / F1**, class-agnostic P / R / F1, **confusion matrix** for localized-but-wrong-class,
  - recall by object size: tiny < 16², small < 32², medium < 96², large,
  - **coverage** = images with ≥ 1 prediction, per-image F1 histogram,
  - **ready** images (per-image F1 ≥ 0.9), **usable** (F1 ≥ 0.7), **effort** per image = FP + FN + wrong class, and
    *effort ratio* = total effort / total GT boxes (1.0 ≈ labeling from scratch),
  - COCO mAP50 and mAP50-95 when predictions have confidences,
  - duplicate check: predicted pairs with IoU ≥ 0.7 left on one image.
- Sanity tests that must pass before Phase 1 ends: GT vs GT = 1.0 everywhere; GT with 10 % boxes removed → recall 0.90;
  GT boxes shifted by 50 % of their width → F1 ≈ 0.
- Also write `tools/oracle.py`, the **simulated human** used by Tracks B–D:
  - `review(image) → accept | reject` : accept if per-image F1 ≥ 0.8;
  - `fix(image)` : replace the image's annotations with GT (`PUT .../annotations`, `source: manual`) and log the
    effort it represents;
  - `label_proposal(proposal) → class | erase` : the class of the GT box with the highest IoU ≥ 0.5, else erase.
  It must use only the API calls the frontend itself makes (read `frontend/js/api.js` and `app.js` to find them).

### 4.3 Platform agent (API + everything except the editor canvas)

Start by running both existing suites and record the baseline (commands in `TESTS.md`: backend in Docker with the
current code mounted, frontend with `python3 tests/frontend/run.py`). Then, on the test instance, test each feature in
`DOCUMENTATION.md` / `FEATURES_USAGE.md`; one row per check in `report/metrics/platform_checklist.md`
(feature, steps, expected, actual, PASS/FAIL):

- **Projects / classes:** create detect, segment, classify; blank name → "Untitled"; duplicate class 409; empty name
  400; palette colors; rename; delete; `order_idx` stability.
- **Upload:** the 200 images (multi-file), an unsupported file (goes to *skipped*), the same file twice (`_1` suffix),
  an EXIF-rotated phone photo, thumbnails exist.
- **Import:** `vd200_gt.zip` into a new detect project **VD-GT** → 200 images, 10 classes in the right order, boxes
  match GT (run `eval.py --source all` = F1 1.0). Also: a zip without `data.yaml` (names become `class0…` — note
  it), a nested `train/images` layout (split pinning), a seg-format zip into a detect project.
- **Merge:** VD-GT into another detect project (classes by name), detect → segment must be refused.
- **Statuses** (`DOCUMENTATION.md` §6.3): every transition (annotated, auto-annotated, review, needs_reannotation,
  empty, approved), Review filter counts, header counts `imgs · labeled · predicted`, `expect_annotations` 409.
- **Versions:** approved-only default (must refuse with nothing approved), splits 70/20/10, pinned splits respected,
  resize stretch/letterbox 640 and 1280, grayscale, augmentation multiplier 3 with all ops; **check that tiny VisDrone
  boxes survive augmentation** (count boxes before/after, look for boxes outside the image or with zero size), YOLO and
  COCO formats (COCO loads in pycocotools, category ids 1-based), zip download, augment preview image.
- **Health:** score, warnings; force leakage (pin one image to train and its copy to val) → leakage warning; a class with
  < 5 labels → warning; `?refresh=true`.
- **Raw export:** zip, `classes.txt`, `data.yaml`, label count = DB count.
- **Training (local):** yolo11n, 3 epochs, imgsz 640, on a VD-GT version (GPU lock); log streaming, plots, stop a
  second run mid-way, delete; the trained weight appears in the model lists; **Test model** on it.
- **SSH training:** if the user has an SSH host, run the smoke test from `FEATURES_USAGE.md` §1; otherwise test only
  validation: password field absent, `host = "-oProxyCommand=…"` rejected, unreachable host gives a readable error.
- **Run a model** with COCO `yolo11n.pt` on a copy of the images: `create_missing_classes` off → only `car`, `bus`,
  `truck`, `bicycle` match (report `person`/`motorcycle` dropped — a class-mapping gap, not necessarily a bug);
  `class_filter`, `replace_existing`, scope *Rejected images*, preview grid, Approve / Approve all. Score it with
  `eval.py` as **Track 0 (zero-shot baseline)**.
- **Active learning:** rank with the Track C weight (later), "Most uncertain first" order = descending uncertainty.
- **Project deletion:** a project with a training job → its `runs/job_<id>/` and weights are gone (check
  `$RUN_DIR/instance-data`).
- **Restart resilience:** start a SAM find-all on 50 images, `docker restart yaap-test` mid-job → job shows failed
  ("Interrupted…"), not running forever; images keep a consistent state.

### 4.4 Annotator agent (UI, headless Chrome) — main bug hunt

Tooling: Chrome is at `/usr/bin/google-chrome`; Playwright is not installed — install it into `$RUN_DIR/venv`
(`pip install playwright`) and launch with `executable_path="/usr/bin/google-chrome"` (no browser download needed).
Record **console errors and page errors on every step** (any one is a finding), take a screenshot on every failure,
and measure timings. Work in a dedicated detect project **VD-UI** (20 images: dense, sparse, night, one > 300 objects,
one 2000×1500) plus a segment project **VD-UI-seg** (10 images).

For every check: do it through the UI, then **verify the stored result through the API** (coordinates in image
pixels, class, kind, source, status). Drawing checks must compute the expected image coordinates from the editor's
view transform at **three zoom levels** (fit, 4×, 16×).

| Area | Checks |
|---|---|
| Load / view | every image loads; fit; zoom wheel around the cursor; pan (H, drag); crosshair; 2000×1500 image sharp at 16× |
| Box tool | draw; draw a **tiny box (6×6 px image space)** at 16× zoom; move; resize with all 8 handles; box dragged past the image edge (clamped?); zero-size click (must not create a box) |
| Dense scene | load GT (import) of the > 300-object image: frame time when panning/zooming, selecting the right box among overlapping ones, handle hit-testing when boxes are 5 px wide, list panel scroll |
| Classes | hotkeys 1–9; **how is the 10th class (motor) chosen?**; change class from the list; delete from the list; class colors |
| Polygon (segment) | add points; close by first vertex / double-click / Enter; drag vertex; click edge inserts vertex; click vertex deletes, never below 3; Esc cancels draft |
| Undo / redo | every operation in `FEATURES_USAGE.md` §13 incl. polygon-draft undo; per-image history after next/prev; history cleared on leaving; ↶/↷ buttons enabled state |
| Autosave | change then wait 0.7 s → saved; **change then go to the next image within 200 ms** (lost edit?); change then close the tab; two quick edits in a row; offline server (stop container) → is the user told? |
| Topstrip | Approve, Reject (toggle), ∅ Mark empty (clears annotations, status `empty`), Prev / Next walking the chosen scope |
| SAM tool (GPU lock) | warm-up hint; point; shift-click negative point; box prompt; chip class picker above the mask (position at zoom/pan, near the image edge); Accept with Enter; simplify slider re-runs the prompt; detect project stores the box, segment stores the polygon; `source: model` |
| Find all like this (E) | box around one car → E → walkthrough: class pick, `1`–`9`, Del/X erase, ←/→ skip, Enter keeps labeled, Esc asks when some are labeled; "· N already labeled"; pan keeps the box out from under the panel; a tiny pedestrian as the example |
| Label found objects | after Track B's generic run: walkthrough, jump to the next image with proposals, Esc pauses, unlabeled ones dropped at Done |
| Review | grid filters + counts, open in editor keeps the filtered order, quick approve / reject / erase on tiles; Quick Review A / R / ⏏ Exit / Esc; the 409 path (a job replaces the image's predictions while it is open) |
| Empty states | annotator with no images in scope → "No images to annotate" + Return to Home |
| Concurrency | edit an image while auto-label processes the same image → job log "changed by hand meanwhile, left alone", your edit kept |
| Robustness | double-click buttons, Back/Forward browser buttons, window resize mid-draw, 2 tabs editing the same image (last write wins? warning?) |

Also run `tests/frontend/run.py` before and after and report any difference.

### 4.5 Track A — SAM 3 *Find all by class*

Project **VD-A** (detect): upload the 200 images, add the 10 classes, then 15 seed images get GT (`PUT` as `manual`)
— they are the only place you may tune.

1. On `seed15` only, compare phrase sets with `conf` 0.25 / 0.4 and `max_overlap` 0.7 / 0.9 (log every run):
   - *names:* the class names as-is (`awning-tricycle` → "awning tricycle"),
   - *descriptions* (≤ 80 chars each), e.g. pedestrian "person walking or standing", people "person sitting or riding",
     bicycle "bicycle", car "car", van "van", truck "truck", tricycle "three-wheeled tricycle", awning-tricycle
     "tricycle with a canopy roof", bus "bus", motor "motorcycle or scooter",
   - your own improved set after looking at the failures.
2. Run the best set on all 185 others in one *Find all* job (scope: unannotated). Record wall time, GPU memory peak,
   OOM retries, job log.
3. Score with `eval.py`. Additionally answer: how many objects are below the 1280-px / 60-px² limits (§1.2), i.e. the
   maximum recall SAM 3 can reach here; which class pairs get confused; did `drop_duplicates` remove real objects in
   crowds (compare pre/post counts in the log).
4. Simulated review with `oracle.review` on all 185 → accepted count, rejected count, effort ratio.
5. Repeat step 2 on a **segment** copy (30 images) to check outlines are saved and editable (≈ 10 points).

### 4.6 Track B — SAM 3 generic + labeling

Project **VD-B** (detect, no classes at first — the card must open in *Generic* mode).

1. Generic nouns: compare "object", "vehicle, person", "car, person, bicycle, motorcycle, truck, bus" on `seed15`
   (class-agnostic recall + proposals per image). Then run the best on all 185.
2. The *Add classes →* flow: no classes → button says Add classes → Classes page → Label them →.
3. Label proposals:
   - with `oracle.label_proposal` for all 185 images (API, the same calls the editor makes),
   - **through the UI** walkthrough on 3 images (coordinate with the Annotator agent; real key presses), and check the
     result equals what the API path produced.
4. Score with `eval.py`. Report the work this needs: proposals to look at, clicks (1 per proposal + 1 per missed GT
   box drawn by hand), vs. drawing all GT boxes from scratch.

### 4.7 Track C — Bootstrap from 15 images

Project **VD-C** (detect): upload the 200, 10 classes, `seed15` gets GT as `manual` / status `annotated`.
Label **2 of the 15 through the editor UI** (box tool, real clicks) instead of the API — that also tests the
annotator under real use; diff them against GT.

1. Round 1 variants (each in its own project copy, GPU lock): `imgsz` 640 (default) and 1280; `epochs` auto; base
   `yolo11n`. Also try `yolo11s` at 1280 if memory allows (OOM is a finding only if the error is unreadable).
2. Record: epochs, training time, GPU memory, the card's warnings ("Few examples of …"), labeling time, the
   "duplicate(s) skipped" counts, images with exactly 300 predictions (the cap).
3. Score each with `eval.py`. Check the card counters, the Train page entry "bootstrap round 1", the registered weight
   name "Bootstrap r1 · yolo11n · 15 imgs".
4. Simulated review loop: `oracle.review` on 40 random round-1 images → Accept the good ones, Reject the rest, then
   `oracle.fix` 10 rejected (as the human would) → **Round 2** (trains on 15 + accepted + fixed, re-labels the rest and
   the rejected) → score the images nobody touched. Repeat once more (Round 3) if Round 2 improved F1 by > 0.03.
5. Stop a round mid-training and mid-labeling: state, logs, images labeled so far keep their predictions.
6. Active learning with the best Track C weight → rank → is "most uncertain first" the images with the lowest F1?
   (Spearman correlation of uncertainty vs per-image F1.)

### 4.8 Track D — hybrid (Phase 3)

Project **VD-D**: Track A's best settings → `oracle.review` → Accept / Reject → `oracle.fix` 15 rejected → Bootstrap
round (imgsz 1280) re-labels everything not accepted → score. Goal: the most images *ready* or *usable* for the
least effort.

---

## 5. What counts as success (the user's goal)

The goal is "auto-label all 200 images or most of them". Report for every track (A, B, C rounds 1–3, D, and the
zero-shot Track 0):

| Metric | Target |
|---|---|
| Coverage — images that got predictions | ≥ 90 % of 185 |
| Usable images — per-image F1 ≥ 0.7 | report the count; "most" = ≥ 50 % |
| Effort ratio — fixes needed / GT boxes | < 0.5 means auto-labeling clearly saves work |
| Overall F1 and mAP50, per-class F1 | report |
| Time and GPU memory | report |

State plainly when a target is missed and why (tiny objects, the 300 cap, class confusion, the downscale limit).
Hypotheses to confirm or reject (don't bend results to them): A is strong on car/bus/truck and weak on
pedestrian/people and the tricycle pair; B finds most vehicles but leaves most of the labeling work; C at 1280 beats
C at 640; D gives the most usable images.

---

## 6. Bug report template (`report/bugs/<agent>-<n>.md`)

```
# <short title>
Severity: blocker | major | minor | cosmetic        Area: annotator | sam | autolabel | bootstrap | import | export | versions | training | review | api | docs
Found by: <agent>   Reproducible: always | sometimes (x/y) | once
Steps: numbered, exact (UI clicks/keys with zoom level, or curl commands)
Expected: …
Actual: …
Evidence: response JSON / log lines / screenshots (paths) / console errors
Suspected cause: file:line + one sentence (read the code, do not change it)
Test idea: which tests/test_*.py or tests/frontend/*.test.html could cover it
```

Also file **docs mismatches** (the docs say X, the app does Y) and **usability gaps** (e.g. no class-name mapping
for pretrained models, no 10th-class hotkey, no `imgsz`/`max_det` control for auto-label) — as severity *minor*,
area *docs* or the feature's area, clearly marked "gap, not a bug".

---

## 7. Final report (`report/REPORT.md`, by the lead)

1. **Verdict** (5 lines): can YAAP auto-label this dataset, which way, how many of 200 images end up usable, how much
   fixing remains.
2. Dataset summary (from `vd200/README.md`).
3. Track comparison table (§5 metrics) + per-class F1 table + recall by object size.
4. Recommended workflow for VisDrone-like data, with the settings that worked (phrases, conf, imgsz, rounds).
5. Bugs: table sorted by severity (title, area, link), then gaps and docs mismatches.
6. Platform checklist summary (PASS / FAIL counts per area) and the two test-suite runs (before / after).
7. What was not tested and why (e.g. SSH without a host).
8. Clean-up done: `yaap-test` stopped and removed, live instance untouched (show `curl :8811/api/projects` before and
   after are identical), `$RUN_DIR` kept for the user.
