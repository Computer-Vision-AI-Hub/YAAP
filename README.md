# YAAP — Yet Another Annotation Platform

**by ComputerVisionAIHub**

A fully **local, open-source** computer-vision annotation platform: upload
datasets, segment objects interactively with **SAM 3** (point or box
prompts), refine labels by hand, auto-label at scale with YOLOv8 / YOLO11 /
YOLO26, RT-DETR (and optionally RF-DETR), augment, version, and export as
either **YOLO** (ultralytics-ready) or **COCO** JSON — then train models on
your own GPU. Nothing ever leaves your machine.

```
Upload → SAM-assisted annotation → Auto-label → Generate version → Export / Train → better models → …
```

## Why local?

YAAP is built for teams (medical imaging, industrial inspection, research…)
that cannot send data to third-party services:

- **No accounts, no telemetry, no cloud.** Storage is a SQLite DB + plain
  folders under `data/`. Delete the folder, and the data is gone.
- **The web UI binds to `127.0.0.1` by default** — not reachable from the network.
- The only optional outbound traffic is ultralytics downloading model weights
  (e.g. `sam3.pt`, `yolo26n.pt`) the first time you use them. For **air-gapped
  operation**, place or upload the `.pt` files manually and nothing is ever
  downloaded.

## SAM-assisted annotation

The `✦` tool in the editor is the heart of YAAP, backed by **SAM 3** only
(point and box prompts):

| Prompt | How |
|---|---|
| **Point** | Click on the object (shift-click adds a *negative* point to carve away background). Every extra click refines the mask. |
| **Box** | Drag a rough rectangle around the object. |

The predicted mask appears as a dashed teal preview. Refine it with more
clicks, then either pick a class from the small chip row that pops up right
next to the mask (labels + accepts in one click), or hit **Accept (↵)** to use
whichever class is active in the sidebar — YAAP converts it to the format your
project actually trains on:

- **Detection project** → the mask's enclosing box, stored as a bounding box.
- **Segmentation project** → a simplified polygon (editable vertex-by-vertex).

So one interaction model produces correctly-formatted labels for both
architectures. Rejected masks (`Esc`) leave no trace.

## Features

| Area | What you get |
|---|---|
| Tasks | Object detection (boxes), instance segmentation (polygons), classification |
| Editor | Canvas with pan/zoom, SAM assist, box + polygon tools, vertex editing, class hotkeys `1-9`, autosave, crosshair guides |
| SAM | SAM 3 point / box prompts, mask→polygon simplification, task-aware conversion, per-mask class picker |
| Auto-label | Batch inference with any ultralytics model — YOLOv8 / YOLO11 / YOLO26, `-seg`, `-cls`, RT-DETR, your uploaded or YAAP-trained `.pt`; optional RF-DETR. Predictions land as editable annotations flagged *for review* |
| Augmentation | albumentations pipeline: flips, rotate, shear, brightness/contrast, hue/sat, blur, noise, CLAHE, cutout, grayscale — with live preview, applied to the train split only |
| Versions | Frozen dataset snapshots: split ratios, preprocessing (resize / stretch / letterbox / grayscale), augmentation multiplier, choice of export format, zip download |
| Export | **YOLO** — detect txt, YOLO-seg polygon txt, or classification folders, with `data.yaml`, directly consumable by `yolo train` / the ultralytics Python API. **COCO** — `_annotations.coco.json` per split (Roboflow-style layout), for RF-DETR / HuggingFace `transformers` / detectron2 / pycocotools. Detect and segment projects can pick either; classify is YOLO-folder only |
| Training | Launch ultralytics training jobs from the UI (subprocess, live log, stop button) for YOLO or RT-DETR. Best weights feed back into auto-labeling. A separate Google Colab tab gives a copyable notebook template for training elsewhere |
| GPU | PyTorch with CUDA — device is auto-detected and shown in the top bar; SAM, auto-label and training all use it |

## Quick start — Docker (recommended)

```bash
git clone https://github.com/ComputerVisionAIHub/YAAP && cd YAAP

# GPU is the default (needs NVIDIA driver + nvidia-container-toolkit)
docker compose up --build

# CPU fallback (no GPU on this machine)
docker compose --profile cpu up --build yaap-cpu
```

Open **http://127.0.0.1:8811**. All state persists in `./data` on the host.

On first start the container pre-downloads the weights listed in the
`PRELOAD_MODELS` env (default: `sam3.pt yolo11n.pt`) into
`./data/models` — after that it runs fully offline. Set `PRELOAD_MODELS: ""`
in `docker-compose.yml` to forbid all downloads and supply `.pt` files yourself.

GPU prerequisites on the host:
1. NVIDIA driver
2. [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   (`sudo apt install nvidia-container-toolkit && sudo systemctl restart docker`)
3. Verify: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

## Quick start — bare Python

```bash
cd YAAP
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1. torch first — pick the wheel matching your CUDA (https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # GPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only

# 2. everything else (includes ultralytics, which provides SAM 3, and timm,
#    which SAM 3's backbone needs)
pip install -r requirements.txt

# 3. optional: RF-DETR support
pip install rfdetr

python run.py          # → http://127.0.0.1:8811
```

> **SAM 3 note.** SAM 3 support ships with recent `ultralytics` releases —
> keep it updated (`pip install -U ultralytics`). `sam3.pt` downloads once on
> first use and is cached under `data/models/` (or `seg_models/`, a dedicated
> bind-mounted directory checked first — handy if you'd rather keep SAM
> weights separate from everything else under `data/`). If it fails to load,
> check your `ultralytics` version first.

## Workflow

1. **New project** — pick the task type. This decides the tools you see and the
   export format, so the dataset always matches the target architecture
   (`yolo11n.pt` ↔ detect, `yolo11n-seg.pt` ↔ segment, `yolo11n-cls.pt` ↔ classify).
2. **Classes** — define them before labeling; their order becomes the YOLO class
   index, so keep it stable across versions.
3. **Upload** — drag & drop; originals are stored untouched, thumbnails are generated.
4. **Annotate** — `S` SAM assist, `B` box, `P` polygon, `V` select, `H`/Space pan,
   `F` fit, `1-9` class, `←/→` image, `↵` accept SAM mask, `Del` remove,
   `Esc` discard. Saves automatically.
5. **Auto-label** — run a pretrained / uploaded / YAAP-trained model over
   unannotated images, then review (model output is marked with confidence).
6. **Generate** — choose splits, preprocessing and augmentations (preview them
   first), set the per-image multiplier, generate a version.
7. **Versions** — download the zip, or note the folder path under
   `data/projects/<id>/versions/v<N>/`.
8. **Train** — pick version + architecture + epochs; watch the live log. The
   resulting `best.pt` shows up in your model list (★) for the next
   auto-label round — a human-in-the-loop flywheel.

Training outside YAAP works too, since every version is standard ultralytics:

```bash
yolo detect train data=data/projects/1/versions/v3/data.yaml model=yolo26s.pt epochs=200 imgsz=640 device=0
```

```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")
model.train(data="data/projects/1/versions/v3/data.yaml", epochs=200, imgsz=640, device=0)
```

## Repository layout

```
YAAP/
├─ run.py                     # python run.py → uvicorn on 127.0.0.1:8811
├─ requirements.txt
├─ Dockerfile                 # targets: cpu | gpu
├─ docker-compose.yml
├─ backend/
│  ├─ app.py                  # FastAPI app + static hosting
│  ├─ config.py  db.py  schemas.py
│  ├─ routers/                # projects, images, annotations, sam, models, datasets, training
│  └─ services/
│     ├─ storage.py           # uploads, thumbnails, project folders
│     ├─ sam.py               # SAM 3 point/box prompting, mask → polygon
│     ├─ inference.py         # ultralytics (YOLO/RT-DETR) + RF-DETR batch auto-label (lazy imports, GPU auto)
│     ├─ augmenter.py         # albumentations; polygons via keypoints
│     ├─ exporter.py          # version generation, YOLO or COCO det/seg/cls export, zip
│     ├─ importer.py          # zip → project import, project → project merge
│     ├─ trainer.py           # subprocess job control
│     └─ train_worker.py      # the actual `ultralytics` training process
├─ frontend/                  # zero-build vanilla JS
│  ├─ index.html  css/yaap.css
│  └─ js/api.js  js/editor.js  js/app.js
└─ data/                      # created at runtime — ALL of your data lives here
   ├─ yaap.db                 # SQLite
   ├─ projects/<id>/{images,thumbs,versions}
   ├─ models/                 # uploaded + cached weights (incl. SAM)
   └─ runs/                   # ultralytics training runs + logs
```

## API

Interactive OpenAPI docs at **http://127.0.0.1:8811/docs** — everything the UI
does is a plain REST call (including SAM prompting at
`POST /api/projects/{pid}/images/{iid}/sam`), so you can script bulk imports,
programmatic SAM labeling, or CI exports.

## Testing

```bash
docker compose exec yaap pytest -v      # or: pytest, if running the bare-Python setup
```

A `pytest` suite in `tests/` covers project/class/image/annotation CRUD and
dataset version generation (both YOLO and COCO layouts) against a throwaway
SQLite DB and data dir (`YAAP_DATA_DIR`, set automatically by
`tests/conftest.py`) — it never touches your real `data/`. It doesn't cover
anything that needs torch/ultralytics actually loaded (SAM, auto-label,
training) — those are exercised manually.

## Notes & limits

- Annotations are stored in pixel coordinates in SQLite and normalized only at
  export time, so re-exporting at different resolutions is lossless.
- SAM masks are simplified to polygons with a configurable tolerance
  (`simplify` in the API request); only the largest connected region per mask
  is kept, which is the right behavior for instance labels.
- Augmentation transforms polygons through keypoints, so only
  polygon-safe geometric ops are exposed (no elastic/perspective warps).
- In a detection project, polygons are converted to enclosing boxes at export;
  in a segmentation project, boxes become 4-point polygons — so mixed labeling
  still exports cleanly.
- `docker-compose.yml` publishes the port on `127.0.0.1` only; change the
  mapping if you deliberately want LAN access, and put a reverse proxy with
  auth in front if you do.
- Google Fonts are referenced by the UI; without internet the app simply falls
  back to system fonts — everything else is served locally.

## License

MIT © ComputerVisionAIHub — do whatever helps you ship datasets.

YAAP's own code is MIT. Its ML backbone (`ultralytics`, powering YOLO,
RT-DETR, and SAM) is **AGPL-3.0**, and Meta's SAM weights carry their own
separate license — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before using this commercially. See [CONTRIBUTING.md](CONTRIBUTING.md) if
you'd like to submit a fix or feature.
