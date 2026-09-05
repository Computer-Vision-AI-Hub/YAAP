"""YAAP configuration — everything lives under DATA_DIR, fully local."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("YAAP_DATA_DIR", BASE_DIR / "data"))
PROJECTS_DIR = DATA_DIR / "projects"
MODELS_DIR = DATA_DIR / "models"          # uploaded / pretrained weights cache
RUNS_DIR = DATA_DIR / "runs"              # ultralytics training runs
DB_PATH = DATA_DIR / "yaap.db"

# SAM weights 
SEG_MODELS_DIR = Path(os.environ.get("SEG_MODELS_DIR", BASE_DIR / "seg_models"))

for d in (DATA_DIR, PROJECTS_DIR, MODELS_DIR, RUNS_DIR, SEG_MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

THUMB_SIZE = 320
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Pretrained aliases shown in the UI.  Ultralytics downloads them on demand.
PRETRAINED_MODELS = {
    "detect": [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
    ],
    "segment": [
        "yolov8n-seg.pt", "yolov8s-seg.pt",
        "yolo11n-seg.pt", "yolo11s-seg.pt",
        "yolo26n-seg.pt", "yolo26s-seg.pt",
    ],
    "classify": [
        "yolov8n-cls.pt", "yolov8s-cls.pt",
        "yolo11n-cls.pt", "yolo11s-cls.pt",
        "yolo26n-cls.pt", "yolo26s-cls.pt",
    ],
}

RFDETR_VARIANTS = ["rfdetr-base", "rfdetr-large"]  # optional, detection only

RTDETR_VARIANTS = ["rtdetr-l.pt", "rtdetr-x.pt"]  # ships inside ultralytics itself, detection only
