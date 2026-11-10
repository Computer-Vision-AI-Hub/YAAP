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
import gc
import threading

import cv2
import numpy as np

from ..config import MODELS_DIR, SEG_MODELS_DIR


class SamError(RuntimeError):
    pass


# ── GPU memory management ─────────────────────────────────────────────
# SAM 3 comes in two incompatible copies here (point/box segmenter = "prompt", and the
# "find all" concept predictor = "concept"), ~2 GB each. Only one lives on the GPU; the
# other is parked in CPU RAM and swapped back in (~1 s, vs ~8 s for a cold load).
# SAM calls are serialized so a request can't park a model another request is using.
_gpu_lock = threading.RLock()
_loaded: dict[str, object] = {}     # kind -> ultralytics object that has been created
FIND_ALL_MAX_SIDE = 1280            # "find all" decodes a downscaled copy (less RAM / transfer)
# What SAM 3 itself sees: its native 1008 px square (ultralytics would otherwise use its generic
# default of 640 → 644, which loses small objects); on CUDA out-of-memory retry once at 644.
FIND_ALL_IMGSZ = (1008, 644)


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def _move(obj, device):
    m = getattr(obj, "model", None)
    if m is not None and hasattr(m, "to"):
        m.to(device)


def _park(kind: str):
    """Move a model to CPU RAM, freeing its GPU memory."""
    if kind in _loaded:
        if kind == "prompt":
            _forget_image(_loaded[kind])     # its cached image encoding lives on the GPU too
        _move(_loaded[kind], "cpu")
        _free_gpu()


def release_gpu():
    """Park every loaded copy in CPU RAM — frees the GPU for a training run."""
    with _gpu_lock:
        for kind in list(_loaded):
            _park(kind)


def _use(kind: str, model_name: str, device):
    """Return `kind`'s model on the GPU, parking the other one."""
    _park("concept" if kind == "prompt" else "prompt")
    obj = _load_sam(model_name) if kind == "prompt" else _load_semantic(model_name, device)
    _loaded[kind] = obj
    _move(obj, device)
    return obj


def warmup(model_name: str = "sam3.pt", device: str = "auto") -> dict:
    """Load both copies once (and run a dummy inference so CUDA kernels are initialised),
    leaving the point/box segmenter active and the concept predictor parked in RAM."""
    if model_name not in SAM_MODELS:
        raise SamError(f"Unknown SAM model '{model_name}'.")
    dev = _device(device)
    blank = np.full((256, 256, 3), 127, dtype=np.uint8)
    with _gpu_lock:
        try:
            pred = _use("concept", model_name, dev)
            pred.args.conf = 0.5
            pred.set_image(blank)
            pred(text=["object"])
            prompt_model = _use("prompt", model_name, dev)
            _forget_image(prompt_model)       # a full call would otherwise reuse a cached image's features
            prompt_model(blank, bboxes=[[10.0, 10.0, 100.0, 100.0]], device=dev, verbose=False)
        except SamError:
            raise
        except Exception as e:
            _free_gpu()
            raise SamError(f"SAM warm-up failed: {e}") from e
        _free_gpu()
    return {"ok": True}


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
@functools.lru_cache(maxsize=1)
def _load_semantic(weights: str, device: str):
    """SAM 3's concept predictor: one box exemplar or a text phrase → every match in the image."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as e:
        raise SamError("'Find all like this' needs a recent ultralytics with SAM 3 "
                       "(`pip install -U ultralytics`).") from e
    for d in (SEG_MODELS_DIR, MODELS_DIR):
        if (d / weights).exists():
            weights = str(d / weights)
            break
    try:
        return SAM3SemanticPredictor(overrides=dict(
            conf=0.25, task="segment", mode="predict", model=weights, save=False, imgsz=FIND_ALL_IMGSZ[0],
            verbose=False, device=device, half=device != "cpu"))
    except Exception as e:
        raise SamError(f"Could not load '{weights}': {e}") from e


def predict_all(image_path: str, model_name: str, box: list[float] | None = None,
                text: str = "", conf: float = 0.25, device: str = "auto",
                simplify: float = 0.008) -> list[dict]:
    """Every instance similar to `box` (an exemplar in THIS image), or matching the
    text phrase. Returns [{polygon, bbox, score, label}] like predict()."""
    if model_name not in SAM_MODELS:
        raise SamError(f"Unknown SAM model '{model_name}'.")
    if box is None and not text.strip():
        raise SamError("Give an exemplar box or a text phrase.")
    if box is not None and len(box) != 4:
        raise SamError("Exemplar box needs [x1, y1, x2, y2].")
    with _gpu_lock:
        for imgsz in FIND_ALL_IMGSZ:                  # on out-of-memory retry once, smaller
            try:
                out = _predict_all(image_path, model_name, box, text, conf, device, simplify, imgsz)
                _free_gpu()
                return out
            except SamError:
                raise
            except Exception as e:
                _free_gpu()
                if not _is_oom(e):
                    hint = (" — install it: pip install ftfy regex git+https://github.com/ultralytics/CLIP.git"
                            if "clip" in str(e).lower() else "")
                    raise SamError(f"SAM inference failed: {e}{hint}") from e
                _park("prompt")
        raise SamError("GPU out of memory — close other GPU jobs (training, auto-label), "
                       "raise the min score, or try again.")


def predict_phrases(image_path: str, model_name: str, phrases: list[str], conf: float = 0.4,
                    device: str = "auto", simplify: float = 0.008) -> dict[str, list[dict]]:
    """Several text phrases on ONE image — the image is encoded once, then each phrase
    is a cheap query. Returns {phrase: [{polygon, bbox, score, label}]}."""
    if model_name not in SAM_MODELS:
        raise SamError(f"Unknown SAM model '{model_name}'.")
    phrases = [p.strip() for p in phrases if p.strip()]
    if not phrases:
        raise SamError("Give at least one text phrase.")
    with _gpu_lock:
        for imgsz in FIND_ALL_IMGSZ:
            try:
                out = _predict_phrases(image_path, model_name, phrases, conf, device, simplify, imgsz)
                _free_gpu()
                return out
            except SamError:
                raise
            except Exception as e:
                _free_gpu()
                if not _is_oom(e):
                    raise SamError(f"SAM inference failed: {e}") from e
                _park("prompt")
        raise SamError("GPU out of memory — close other GPU jobs (training, auto-label), "
                       "raise the min score, or try again.")


def _predict_phrases(image_path, model_name, phrases, conf, device, simplify, imgsz):
    import cv2
    pred = _use("concept", model_name, _device(device))
    pred.args.conf = conf
    pred.args.imgsz = imgsz              # set_image() re-reads it for every image
    img = cv2.imread(image_path)
    if img is None:
        raise SamError("Could not read the image.")
    f = min(1.0, FIND_ALL_MAX_SIDE / max(img.shape[:2]))
    if f < 1.0:
        img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    pred.set_image(img)
    out = {}
    for ph in phrases:
        found = _collect(pred(text=[ph]), simplify)
        if f < 1.0:
            for r in found:
                r["polygon"] = [[x / f, y / f] for x, y in r["polygon"]]
                r["bbox"] = _poly_bbox(r["polygon"])
        out[ph] = found
    return out


def _predict_all(image_path, model_name, box, text, conf, device, simplify, imgsz):
    import cv2
    pred = _use("concept", model_name, _device(device))
    pred.args.conf = conf
    pred.args.imgsz = imgsz
    img = cv2.imread(image_path)
    if img is None:
        raise SamError("Could not read the image.")
    f = min(1.0, FIND_ALL_MAX_SIDE / max(img.shape[:2]))
    if f < 1.0:
        img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    pred.set_image(img)
    results = pred(bboxes=[[float(v) * f for v in box]]) if box is not None else pred(text=[text.strip()])
    out = _collect(results, simplify)
    if f < 1.0:                                     # back to the original image's pixel coordinates
        for r in out:
            r["polygon"] = [[x / f, y / f] for x, y in r["polygon"]]
            r["bbox"] = _poly_bbox(r["polygon"])
    return out


def predict(image_path: str, model_name: str, prompt: dict, device: str = "auto",
            simplify: float = 0.008) -> list[dict]:
    """prompt: {type: 'point'|'box', points:[[x,y]..], labels:[1,0..], box:[x1,y1,x2,y2]}
    Returns [{polygon, bbox, score, label}] — polygons in pixel coords."""
    ptype = prompt.get("type")
    if model_name not in SAM_MODELS:   # never load a client-chosen path (pickle)
        raise SamError(f"Unknown SAM model '{model_name}'.")
    with _gpu_lock:
        return _predict_prompt(image_path, model_name, prompt, ptype, device, simplify)


# Click/box prompts on one image reuse its encoding: encoding a photo takes ~2.2 s on an RTX 4050, a prompt on
# the encoded features ~0.03 s — so only the first click on an image waits. ultralytics' predictor silently
# reuses whatever features it holds, so they are dropped whenever the image (or its file) changes.
_encoded: dict = {"key": None}


def _forget_image(model):
    pred = getattr(model, "predictor", None)
    if pred is not None and hasattr(pred, "reset_image"):
        pred.reset_image()
    _encoded["key"] = None


def _prompt_cached(model, image_path, kwargs):
    import os
    pred = getattr(model, "predictor", None)
    if pred is None or not hasattr(pred, "set_image"):     # predictor not set up yet (no warm-up): plain call
        _forget_image(model)
        return model(image_path, **kwargs)
    key = (image_path, os.path.getmtime(image_path), str(kwargs.get("device")))
    if _encoded["key"] != key:
        _forget_image(model)
        pred.set_image(image_path)
        _encoded["key"] = key
    prompts = {k: v for k, v in kwargs.items() if k in ("points", "labels", "bboxes")}
    return pred(**prompts)


def _predict_prompt(image_path, model_name, prompt, ptype, device, simplify):
    model = _use("prompt", model_name, _device(device))
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
        results = _prompt_cached(model, image_path, kwargs)
    except Exception as e:
        _forget_image(model)
        if _is_oom(e):
            _free_gpu()
            raise SamError("GPU out of memory — close other GPU jobs (training, auto-label) and try again.") from e
        raise SamError(f"SAM inference failed: {e}") from e
    out = _collect(results, simplify)
    return out


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
