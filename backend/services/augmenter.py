"""Image preprocessing + augmentation built on albumentations.

Geometric ops are restricted to transforms that keep polygons valid
(we route polygon vertices through keypoint_params).
"""
from __future__ import annotations

import copy
import random

import cv2
import numpy as np


def _A():
    try:
        import albumentations as A
        return A
    except ImportError as e:
        raise RuntimeError("albumentations is not installed — `pip install albumentations`.") from e


# ── pipeline builders ────────────────────────────────────────────────
def build_preprocess(cfg: dict):
    """cfg: {resize:int|0, mode:'stretch'|'letterbox', grayscale:bool}"""
    A = _A()
    t = []
    size = int(cfg.get("resize") or 0)
    if size > 0:
        if cfg.get("mode") == "letterbox":
            t += [A.LongestMaxSize(max_size=size),
                  A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114))]
        else:
            t += [A.Resize(size, size)]
    if cfg.get("grayscale"):
        t += [A.ToGray(p=1.0)]
    return t


def build_augment(ops: dict):
    """ops: dict of toggles/values coming from the UI."""
    A = _A()
    t = []
    if ops.get("hflip"):
        t.append(A.HorizontalFlip(p=0.5))
    if ops.get("vflip"):
        t.append(A.VerticalFlip(p=0.5))
    if ops.get("rotate90"):
        t.append(A.RandomRotate90(p=0.5))
    deg = float(ops.get("rotate", 0) or 0)
    if deg > 0:
        t.append(A.Rotate(limit=deg, border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.7))
    sh = float(ops.get("shear", 0) or 0)
    if sh > 0:
        t.append(A.Affine(shear=(-sh, sh), border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.5))
    br = float(ops.get("brightness", 0) or 0)
    ct = float(ops.get("contrast", 0) or 0)
    if br > 0 or ct > 0:
        t.append(A.RandomBrightnessContrast(brightness_limit=br, contrast_limit=ct, p=0.7))
    hue = float(ops.get("hue", 0) or 0)
    sat = float(ops.get("saturation", 0) or 0)
    if hue > 0 or sat > 0:
        t.append(A.HueSaturationValue(hue_shift_limit=int(hue * 180), sat_shift_limit=int(sat * 255),
                                      val_shift_limit=0, p=0.7))
    if float(ops.get("blur", 0) or 0) > 0:
        t.append(A.GaussianBlur(blur_limit=(3, int(3 + ops["blur"] * 10) | 1), p=0.4))
    if float(ops.get("noise", 0) or 0) > 0:
        t.append(A.GaussNoise(std_range=(0.02, 0.02 + 0.15 * float(ops["noise"])), p=0.4))
    if ops.get("clahe"):
        t.append(A.CLAHE(p=0.4))
    if ops.get("gray_p"):
        t.append(A.ToGray(p=0.15))
    cut = float(ops.get("cutout", 0) or 0)
    if cut > 0:
        t.append(A.CoarseDropout(num_holes_range=(2, 8),
                                 hole_height_range=(8, int(64 * cut) + 9),
                                 hole_width_range=(8, int(64 * cut) + 9),
                                 fill=114, p=0.5))
    return t


def compose(transforms, with_targets: bool):
    A = _A()
    if not with_targets:
        return A.Compose(transforms)
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_ids"],
                                 min_visibility=0.2, min_area=4),
        keypoint_params=A.KeypointParams(format="xy", label_fields=["kp_ids"],
                                         remove_invisible=False),
    )


# ── application ──────────────────────────────────────────────────────
def apply(pipeline, image: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """annotations: YAAP dicts (pixel coords). Returns transformed copies."""
    h, w = image.shape[:2]
    bboxes, bbox_ids = [], []
    keypoints, kp_ids = [], []   # kp_ids: (ann_index, point_index)

    for i, ann in enumerate(annotations):
        if ann["kind"] == "bbox":
            d = ann["data"]
            x1, y1 = max(0.0, d["x"]), max(0.0, d["y"])
            x2, y2 = min(float(w), d["x"] + d["w"]), min(float(h), d["y"] + d["h"])
            if x2 - x1 > 1 and y2 - y1 > 1:
                bboxes.append([x1, y1, x2, y2]); bbox_ids.append(i)
        elif ann["kind"] == "polygon":
            for j, (px, py) in enumerate(ann["data"]["points"]):
                keypoints.append((min(max(px, 0), w - 1), min(max(py, 0), h - 1)))
                kp_ids.append(i * 100000 + j)      # scalar id: ann_idx * 1e5 + point_idx

    res = pipeline(image=image, bboxes=bboxes, bbox_ids=bbox_ids,
                   keypoints=keypoints, kp_ids=kp_ids)
    out_img = res["image"]
    oh, ow = out_img.shape[:2]
    out_anns: list[dict] = []

    for (x1, y1, x2, y2), idx in zip(res["bboxes"], res["bbox_ids"]):
        a = copy.deepcopy(annotations[int(idx)])
        a["data"] = {"x": float(x1), "y": float(y1), "w": float(x2 - x1), "h": float(y2 - y1)}
        out_anns.append(a)

    poly_pts: dict[int, dict[int, tuple]] = {}
    for (px, py), kid in zip(res["keypoints"], res["kp_ids"]):
        kid = int(kid)
        ai, pj = kid // 100000, kid % 100000
        poly_pts.setdefault(ai, {})[pj] = (float(px), float(py))
    for ai, pts in poly_pts.items():
        ordered = [pts[k] for k in sorted(pts)]
        clipped = [[float(min(max(px, 0), ow - 1)), float(min(max(py, 0), oh - 1))] for px, py in ordered]
        # drop polygons that collapsed outside the frame
        xs = [p[0] for p in clipped]; ys = [p[1] for p in clipped]
        if len(clipped) >= 3 and (max(xs) - min(xs)) > 2 and (max(ys) - min(ys)) > 2:
            a = copy.deepcopy(annotations[ai])
            a["data"] = {"points": clipped}
            out_anns.append(a)

    for ann in annotations:                      # classification tags pass through
        if ann["kind"] == "classification":
            out_anns.append(copy.deepcopy(ann))
    return out_img, out_anns


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**31))
