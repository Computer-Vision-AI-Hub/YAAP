"""Model-assisted labeling.

"""
from __future__ import annotations

import functools
from pathlib import Path

from ..config import MODELS_DIR, PRETRAINED_MODELS, RFDETR_VARIANTS, RTDETR_VARIANTS

KNOWN_ALIASES = frozenset([m for ms in PRETRAINED_MODELS.values() for m in ms] + RTDETR_VARIANTS)


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
        try:
            wid = int(model_name.split(":", 1)[1])
        except ValueError:
            return False   # resolve_weights() reports the bad reference
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
    """'weight:<id>' → stored path, or a known pretrained alias. Anything else is
    refused: .pt files are pickles, so loading a client-chosen path runs code."""
    if model_name.startswith("weight:") and db is not None:
        from ..db import ModelWeight
        try:
            wid = int(model_name.split(":", 1)[1])
        except ValueError:
            raise InferenceError(f"Invalid weight reference '{model_name}'.")
        w = db.get(ModelWeight, wid)
        if not w or not Path(w.path).exists():
            raise InferenceError(f"Stored weight #{wid} not found on disk.")
        return w.path
    if model_name not in KNOWN_ALIASES:
        raise InferenceError(f"Unknown model '{model_name}' — pick a listed model or upload weights.")
    use_models_dir()
    return alias_path(model_name)


def alias_path(name: str) -> str:
    """A pretrained alias → its file in MODELS_DIR: used as-is when preloaded (PRELOAD_MODELS / start.sh put
    it there), otherwise ultralytics downloads it to exactly that path. A bare alias would be looked up in
    the working directory only and re-downloaded there (lost on every rebuild; fails offline)."""
    return str(MODELS_DIR / name) if name in KNOWN_ALIASES else name


def use_models_dir():
    """Keep ultralytics' settings in MODELS_DIR and let it find preloaded helper weights there too
    (e.g. the yolo26n.pt its AMP check loads by bare name)."""
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR))
    try:
        from ultralytics.utils import SETTINGS
        if Path(str(SETTINGS.get("weights_dir", ""))).resolve() != MODELS_DIR.resolve():
            SETTINGS.update(weights_dir=str(MODELS_DIR))
    except Exception:          # no ultralytics / read-only settings file: lookups just stay as they were
        pass


# ── prediction → YAAP annotation dicts ───────────────────────────────
def predict(model_name: str, image_path: str, task: str, conf: float, iou: float,
            device: str, db=None) -> list[dict]:
    """Returns [{class_name, kind, data, confidence}, ...] in pixel coordinates."""
    device = resolve_device(device)

    if model_name in RFDETR_VARIANTS:
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
            if len(poly) < 3:
                continue
            pts = simplify_polygon(poly.tolist())
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


def simplify_polygon(points, simplify: float = 0.008) -> list[list[float]]:
    """A mask outline (often hundreds of points) → an editable polygon, the same way SAM's
    outlines are simplified (Douglas-Peucker, tolerance = simplify × perimeter)."""
    import cv2
    import numpy as np
    c = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(c, max(1.0, simplify * cv2.arcLength(c, True)), True).reshape(-1, 2)
    pts = approx if len(approx) >= 3 else c.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


UNCERTAINTY_MIN_CONF = 0.25


def detection_uncertainty(confidences: list[float]) -> float:
    """Per-image uncertainty for detect/segment: 1 − the mean confidence of the image's detections
    (those ≥ 0.25 — below that is background noise). Nothing detected scores 0.5 — the model neither
    claims nor rules anything out.
    The earlier rule (the single most ambiguous detection decides) saturated on busy images: with 40+
    objects one detection is always near 0.5, so 62 % of VisDrone images scored ≥ 0.95 and the ranking
    didn't correlate with how wrong the model was (Spearman +0.12 vs per-image F1); the mean gives −0.64."""
    cs = [c for c in confidences if c >= UNCERTAINTY_MIN_CONF]
    if not cs:
        return 0.5
    return 1.0 - sum(cs) / len(cs)


def uncertainty(model_name: str, image_path: str, task: str, device: str, db=None) -> float:
    """0 (model is sure) … 1 (model is torn) for one image — used to rank unlabeled
    images for annotation. Detection runs at a very low conf so borderline
    objects are visible; classify uses normalized entropy of the class probabilities."""
    if task == "classify":
        import math
        device = resolve_device(device)
        loader = _load_yolo
        r = loader(resolve_weights(model_name, db)).predict(
            source=image_path, device=device, verbose=False)[0]
        if r.probs is None:
            raise InferenceError("This model does not output classification probs — pick a *-cls model.")
        pr = [float(v) for v in r.probs.data.tolist()]
        h = -sum(v * math.log(v) for v in pr if v > 0)
        return h / math.log(len(pr)) if len(pr) > 1 else 0.0
    preds = predict(model_name, image_path, task, 0.05, 0.5, device, db)
    return detection_uncertainty([p["confidence"] for p in preds])


def _predict_rfdetr(variant: str, image_path: str, conf: float) -> list[dict]:
    from PIL import Image, ImageOps
    model, names = _load_rfdetr(variant)
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")   # same frame as YOLO/SAM (cv2)
    det = model.predict(im, threshold=conf)
    out = []
    for (x1, y1, x2, y2), cid, score in zip(det.xyxy.tolist(), det.class_id.tolist(), det.confidence.tolist()):
        label = names.get(int(cid), f"class_{int(cid)}") if isinstance(names, dict) else f"class_{int(cid)}"
        out.append({"class_name": str(label), "kind": "bbox",
                    "data": {"x": float(x1), "y": float(y1), "w": float(x2 - x1), "h": float(y2 - y1)},
                    "confidence": float(score)})
    return out
