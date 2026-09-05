"""Model-assisted labeling.

"""
from __future__ import annotations

import functools
from pathlib import Path

from ..config import MODELS_DIR


class InferenceError(RuntimeError):
    pass


# ── device helpers ───────────────────────────────────────────────────
def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False, "devices": [], "message": "PyTorch is not installed."}
    devices = []
    if torch.cuda.is_available():
        devices = [f"cuda:{i} — {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "devices": devices, "message": ""}


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── model loading (cached) ───────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def _load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise InferenceError("ultralytics is not installed — run `pip install ultralytics`.") from e
    try:
        return YOLO(weights)
    except Exception as e:
        raise InferenceError(f"Could not load model '{weights}': {e}") from e


@functools.lru_cache(maxsize=4)
def _load_rtdetr(weights: str):
    try:
        from ultralytics import RTDETR
    except ImportError as e:
        raise InferenceError("ultralytics is not installed — run `pip install ultralytics`.") from e
    try:
        return RTDETR(weights)
    except Exception as e:
        raise InferenceError(f"Could not load model '{weights}': {e}") from e


def is_rtdetr_arch(model_name: str, db=None) -> bool:
    """Returns True if the model is an RT-DETR architecture (vs YOLO)."""
    if model_name.startswith("weight:") and db is not None:
        from ..db import ModelWeight
        wid = int(model_name.split(":", 1)[1])
        w = db.get(ModelWeight, wid)
        return bool(w and "rtdetr" in (w.arch or "").lower())
    return "rtdetr" in model_name.lower()


@functools.lru_cache(maxsize=2)
def _load_rfdetr(variant: str):
    try:
        from rfdetr import RFDETRBase, RFDETRLarge  # type: ignore
    except ImportError as e:
        raise InferenceError("rfdetr is not installed — run `pip install rfdetr` to enable RF-DETR.") from e
    model = RFDETRLarge() if "large" in variant else RFDETRBase()
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES  # type: ignore
        names = COCO_CLASSES
    except Exception:
        names = {}
    return model, names


def resolve_weights(model_name: str, db=None) -> str:
    """'weight:<id>' → stored path, otherwise treat as ultralytics alias/path."""
    if model_name.startswith("weight:") and db is not None:
        from ..db import ModelWeight
        wid = int(model_name.split(":", 1)[1])
        w = db.get(ModelWeight, wid)
        if not w or not Path(w.path).exists():
            raise InferenceError(f"Stored weight #{wid} not found on disk.")
        return w.path
    # keep ultralytics' auto-download inside our data dir
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR))
    return model_name


# ── prediction → YAAP annotation dicts ───────────────────────────────
def predict(model_name: str, image_path: str, task: str, conf: float, iou: float,
            device: str, db=None) -> list[dict]:
    """Returns [{class_name, kind, data, confidence}, ...] in pixel coordinates."""
    device = resolve_device(device)

    if model_name.startswith("rfdetr"):
        return _predict_rfdetr(model_name, image_path, conf)

    loader = _load_rtdetr if is_rtdetr_arch(model_name, db) else _load_yolo
    model = loader(resolve_weights(model_name, db))
    results = model.predict(source=image_path, conf=conf, iou=iou, device=device, verbose=False)
    r = results[0]
    names = r.names
    out: list[dict] = []

    if task == "classify":
        if r.probs is None:
            raise InferenceError("This model does not output classification probs — pick a *-cls model.")
        top = int(r.probs.top1)
        out.append({"class_name": names[top], "kind": "classification",
                    "data": {}, "confidence": float(r.probs.top1conf)})
        return out

    if task == "segment" and r.masks is not None:
        for poly, box in zip(r.masks.xy, r.boxes):
            pts = [[float(x), float(y)] for x, y in poly.tolist()]
            if len(pts) < 3:
                continue
            out.append({"class_name": names[int(box.cls)], "kind": "polygon",
                        "data": {"points": pts}, "confidence": float(box.conf)})
        return out

    if r.boxes is None:
        return out
    for box in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        out.append({"class_name": names[int(box.cls)], "kind": "bbox",
                    "data": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "confidence": float(box.conf)})
    return out


def _predict_rfdetr(variant: str, image_path: str, conf: float) -> list[dict]:
    from PIL import Image
    model, names = _load_rfdetr(variant)
    im = Image.open(image_path).convert("RGB")
    det = model.predict(im, threshold=conf)
    out = []
    for (x1, y1, x2, y2), cid, score in zip(det.xyxy.tolist(), det.class_id.tolist(), det.confidence.tolist()):
        label = names.get(int(cid), f"class_{int(cid)}") if isinstance(names, dict) else f"class_{int(cid)}"
        out.append({"class_name": str(label), "kind": "bbox",
                    "data": {"x": float(x1), "y": float(y1), "w": float(x2 - x1), "h": float(y2 - y1)},
                    "confidence": float(score)})
    return out
