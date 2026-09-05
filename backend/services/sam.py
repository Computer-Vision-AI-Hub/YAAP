"""SAM-assisted annotation — the core interactive segmenter of YAAP.

Prompts supported:
    point(s)  — positive / negative clicks
    box       — rough rectangle around the object


Every mask is converted to a simplified polygon in image pixel coordinates so
the frontend can hand it to the editor as a normal annotation. Conversion to
bounding boxes (for detection projects) happens downstream from the polygon.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np

from ..config import MODELS_DIR, SEG_MODELS_DIR


class SamError(RuntimeError):
    pass


# name → (weights, description)
SAM_MODELS = {
    "sam3.pt": {"label": "SAM 3 — points · bbox"},
}


def list_models() -> dict:
    try:
        import ultralytics  # noqa: F401
        available = True
        version = ultralytics.__version__
    except ImportError:
        available, version = False, ""
    return {"models": [{"name": k, **v} for k, v in SAM_MODELS.items()],
            "ultralytics": available, "ultralytics_version": version}


@functools.lru_cache(maxsize=2)
def _load_sam(weights: str):
    try:
        from ultralytics import SAM
    except ImportError as e:
        raise SamError("ultralytics is not installed — `pip install ultralytics`.") from e
    for d in (SEG_MODELS_DIR, MODELS_DIR):    # dedicated seg_models/ dir wins, then data/models/
        local = d / weights
        if local.exists():                    # preloaded / uploaded → fully offline
            weights = str(local)
            break
    try:
        return SAM(weights)
    except Exception as e:
        raise SamError(
            f"Could not load '{weights}': {e}. "
            "SAM 3 needs a recent ultralytics (`pip install -U ultralytics`); "
            "weights download once, then run offline.") from e


def _device(device: str):
    if device and device != "auto":
        return device
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── mask → polygon ───────────────────────────────────────────────────
def mask_to_polygons(mask: np.ndarray, simplify: float = 0.01,
                     min_area: float = 60.0) -> list[list[list[float]]]:
    """Binary mask → list of simplified polygons [[x,y], ...]."""
    m = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = max(1.0, simplify * cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([[float(x), float(y)] for x, y in approx])
    polys.sort(key=lambda p: -cv2.contourArea(np.array(p, dtype=np.float32)))
    return polys


def _poly_bbox(poly) -> dict:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


# ── prediction ───────────────────────────────────────────────────────
def predict(image_path: str, model_name: str, prompt: dict, device: str = "auto",
            simplify: float = 0.008) -> list[dict]:
    """prompt: {type: 'point'|'box', points:[[x,y]..], labels:[1,0..], box:[x1,y1,x2,y2]}
    Returns [{polygon, bbox, score, label}] — polygons in pixel coords."""
    ptype = prompt.get("type")
    model = _load_sam(model_name)
    kwargs = {"device": _device(device), "verbose": False}
    if ptype == "point":
        pts = prompt.get("points") or []
        if not pts:
            raise SamError("Point prompt needs at least one click.")
        labels = [int(v) for v in (prompt.get("labels") or [1] * len(pts))]
        kwargs["points"] = [[[float(x), float(y)] for x, y in pts]]
        kwargs["labels"] = [labels]
    elif ptype == "box":
        box = prompt.get("box")
        if not box or len(box) != 4:
            raise SamError("Box prompt needs [x1, y1, x2, y2].")
        kwargs["bboxes"] = [[float(v) for v in box]]
    else:
        raise SamError(f"Unknown prompt type '{ptype}'.")

    try:
        results = model(image_path, **kwargs)
    except Exception as e:
        raise SamError(f"SAM inference failed: {e}") from e
    return _collect(results, simplify)


def _collect(results, simplify: float, default_label: str = "") -> list[dict]:
    out = []
    for r in results:
        if r.masks is None:
            continue
        try:
            data = r.masks.data.cpu().numpy()
        except Exception:
            data = np.asarray(r.masks.data)
        confs = []
        if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes):
            try:
                confs = [float(c) for c in r.boxes.conf.tolist()]
            except Exception:
                confs = []
        for i, m in enumerate(data):
            polys = mask_to_polygons(m, simplify)
            if not polys:
                continue
            poly = polys[0]                      # largest region per mask
            out.append({"polygon": poly, "bbox": _poly_bbox(poly),
                        "score": confs[i] if i < len(confs) else 1.0,
                        "label": default_label})
    return out
