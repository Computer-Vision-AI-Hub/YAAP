"""Dataset version generation.

detect/segment tasks always get the ultralytics-ready YOLO layout — needed
for in-platform YOLO/RT-DETR training regardless of `config["format"]`:
    v<id>/
      data.yaml
      train/images  train/labels   (+ val/, test/)
classify tasks always use this folder layout instead (no format choice):
    v<id>/
      train/<class_name>/*.jpg     (+ val/, test/)

`config["format"] == "coco"` adds a second, parallel layout alongside the
YOLO one — Roboflow-style, matching what most COCO-consuming tools
(pycocotools, HuggingFace transformers, detectron2) expect out of the box:
    v<id>/
      train/_annotations.coco.json  train/*.jpg   (+ val/, test/)

So a "coco" version is trainable in-platform (via the YOLO layout it also
contains) *and* exportable for external COCO-based tooling — pick "coco"
whenever you might want either, not just when you're leaving the platform.

A .zip sits next to the folder for one-click download either way.
"""
from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

from ..db import Annotation, DatasetVersion, ImageAsset, LabelClass, Project
from . import augmenter, storage


def _ann_dicts(image: ImageAsset) -> list[dict]:
    out = []
    for a in image.annotations:
        out.append({"kind": a.kind, "class_id": a.class_id, "data": a.data_obj,
                    "confidence": a.confidence})
    return out


def _split_images(images, splits: dict[str, float], seed=1337):
    """Respect manual pins (image.split), randomize the rest by ratio."""
    rng = random.Random(seed)
    pinned = {s: [im for im in images if im.split == s] for s in ("train", "val", "test")}
    rest = [im for im in images if im.split not in ("train", "val", "test")]
    rng.shuffle(rest)
    total = sum(max(splits.get(s, 0), 0) for s in ("train", "val", "test")) or 1.0
    n = len(rest)
    n_train = round(n * splits.get("train", 0) / total)
    n_val = round(n * splits.get("val", 0) / total)
    pinned["train"] += rest[:n_train]
    pinned["val"] += rest[n_train:n_train + n_val]
    pinned["test"] += rest[n_train + n_val:]
    return pinned


def _write_label_txt(path: Path, anns: list[dict], class_idx: dict[int, int], w: int, h: int, task: str):
    lines = []
    for a in anns:
        ci = class_idx.get(a["class_id"])
        if ci is None:
            continue
        if task == "detect" and a["kind"] in ("bbox", "polygon"):
            if a["kind"] == "polygon":                      # polygon → enclosing box
                xs = [p[0] for p in a["data"]["points"]]; ys = [p[1] for p in a["data"]["points"]]
                x, y, bw, bh = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
            else:
                d = a["data"]; x, y, bw, bh = d["x"], d["y"], d["w"], d["h"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{ci} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        elif task == "segment" and a["kind"] == "polygon":
            flat = " ".join(f"{px / w:.6f} {py / h:.6f}" for px, py in a["data"]["points"])
            lines.append(f"{ci} {flat}")
        elif task == "segment" and a["kind"] == "bbox":     # box → rectangle polygon
            d = a["data"]
            pts = [(d["x"], d["y"]), (d["x"] + d["w"], d["y"]),
                   (d["x"] + d["w"], d["y"] + d["h"]), (d["x"], d["y"] + d["h"])]
            flat = " ".join(f"{px / w:.6f} {py / h:.6f}" for px, py in pts)
            lines.append(f"{ci} {flat}")
    path.write_text("\n".join(lines))


def _coco_ann(a: dict, class_idx: dict[int, int]) -> dict | None:
    """One YAAP annotation dict → one COCO annotation dict (sans id/image_id,
    filled in by the caller). category_id is 1-based, the COCO convention."""
    ci = class_idx.get(a["class_id"])
    if ci is None:
        return None
    if a["kind"] == "bbox":
        d = a["data"]
        bbox = [d["x"], d["y"], d["w"], d["h"]]
        seg: list = []
    elif a["kind"] == "polygon":
        pts = a["data"]["points"]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        seg = [[coord for p in pts for coord in p]]
    else:
        return None
    return {"category_id": ci + 1, "bbox": [round(v, 2) for v in bbox],
            "area": round(bbox[2] * bbox[3], 2), "iscrowd": 0, "segmentation": seg}


def _write_coco_json(path: Path, entries: list[tuple[str, int, int, list[dict]]],
                     classes: list[LabelClass], class_idx: dict[int, int]):
    """Roboflow-style layout: one `_annotations.coco.json` per split, sitting
    right alongside that split's images (no images/labels subfolders) — the
    convention most COCO-consuming tools (pycocotools, HF transformers,
    detectron2) expect out of the box."""
    categories = [{"id": i + 1, "name": c.name, "supercategory": "none"} for i, c in enumerate(classes)]
    images_j, anns_j = [], []
    ann_id = 1
    for img_id, (stem, w, h, anns) in enumerate(entries, start=1):
        images_j.append({"id": img_id, "file_name": f"{stem}.jpg", "width": w, "height": h})
        for a in anns:
            rec = _coco_ann(a, class_idx)
            if rec is None:
                continue
            rec["id"] = ann_id
            rec["image_id"] = img_id
            anns_j.append(rec)
            ann_id += 1
    path.write_text(json.dumps({"images": images_j, "annotations": anns_j, "categories": categories}))


def generate(db, project: Project, version: DatasetVersion) -> dict:
    cfg = json.loads(version.config)
    task = project.task_type
    fmt = cfg.get("format", "yolo")
    if fmt == "coco" and task == "classify":
        raise RuntimeError("COCO format doesn't support classification projects — use YOLO export instead.")
    classes: list[LabelClass] = list(project.classes)
    class_idx = {c.id: i for i, c in enumerate(classes)}
    class_name = {c.id: c.name for c in classes}

    images = [im for im in project.images
              if cfg.get("include_unannotated") or im.annotations]
    if cfg.get("approved_only", True):
        images = [im for im in images if im.status == "approved"]
    if not images:
        raise RuntimeError(
            "No approved images to export — approve images in the Review tab first, "
            "or turn off \"Only approved images\" below." if cfg.get("approved_only", True)
            else "No annotated images to export — annotate something first.")

    out_dir = storage.project_dir(project.id) / "versions" / f"v{version.id}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pre = augmenter.compose(augmenter.build_preprocess(cfg.get("preprocess", {})), with_targets=True)
    aug_ops = cfg.get("augment", {}).get("ops", {})
    multiplier = max(1, int(cfg.get("augment", {}).get("multiplier", 1)))
    aug = augmenter.compose(augmenter.build_preprocess(cfg.get("preprocess", {})) +
                            augmenter.build_augment(aug_ops), with_targets=True) \
        if multiplier > 1 and aug_ops else None

    split_map = _split_images(images, cfg.get("splits", {}))
    counts = {s: 0 for s in split_map}                       # files written (incl. augmented copies)
    source_counts = {s: len(ims) for s, ims in split_map.items()}  # distinct source images
    coco_entries: dict[str, list] = {s: [] for s in split_map}     # coco only: (stem, w, h, anns)
    ann_count = 0
    augmenter.seed_everything(1337)

    for split, ims in split_map.items():
        if not ims:
            continue
        if task == "classify":
            (out_dir / split).mkdir(parents=True, exist_ok=True)
        else:
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        for im in ims:
            src = storage.image_path(project.id, im.filename)
            if not src.exists():
                continue
            img = augmenter.load_bgr(str(src))
            anns = _ann_dicts(im)
            variants = [(pre, 0)]
            if aug is not None and split == "train":     # augment training split only
                variants += [(aug, k + 1) for k in range(multiplier - 1)]

            for pipeline, k in variants:
                try:
                    out_img, out_anns = augmenter.apply(pipeline, img, anns)
                except Exception:
                    continue
                stem = f"{Path(im.filename).stem}_{im.id}" + (f"_aug{k}" if k else "")
                if task == "classify":
                    tag = next((a for a in out_anns if a["kind"] == "classification"), None)
                    cname = class_name.get(tag["class_id"], "_unlabeled") if tag else "_unlabeled"
                    cdir = out_dir / split / cname
                    cdir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cdir / f"{stem}.jpg"), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    oh, ow = out_img.shape[:2]
                    cv2.imwrite(str(out_dir / split / "images" / f"{stem}.jpg"), out_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    _write_label_txt(out_dir / split / "labels" / f"{stem}.txt",
                                     out_anns, class_idx, ow, oh, task)
                    ann_count += len(out_anns)
                    if fmt == "coco":                     # parallel Roboflow-style layout
                        cv2.imwrite(str(out_dir / split / f"{stem}.jpg"), out_img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        coco_entries[split].append((stem, ow, oh, out_anns))
                counts[split] += 1

    if task != "classify":
        data_yaml = {
            "path": str(out_dir.resolve()),
            "train": "train/images", "val": "val/images",
            "names": {i: c.name for i, c in enumerate(classes)},
        }
        if counts.get("test"):
            data_yaml["test"] = "test/images"
        if not counts.get("val") and counts.get("train"):
            data_yaml["val"] = "train/images"             # ultralytics needs a val set
        (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    if fmt == "coco":
        for split, entries in coco_entries.items():
            if entries:
                _write_coco_json(out_dir / split / "_annotations.coco.json", entries, classes, class_idx)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir.parent))

    return {"images": counts, "source_images": source_counts,
            "source_images_total": sum(source_counts.values()), "format": fmt,
            "annotations": ann_count, "classes": [c.name for c in classes], "path": str(out_dir)}


def version_dir(project_id: int, version_id: int) -> Path:
    return storage.project_dir(project_id) / "versions" / f"v{version_id}"


def export_raw(project: Project, approved_only: bool = False) -> Path:
    """Pack the project's current images + annotations as-is — no split,
    no preprocessing, no augmentation, no DatasetVersion row. Rebuilt fresh
    on every call so it always reflects the latest annotations."""
    classes: list[LabelClass] = list(project.classes)
    class_idx = {c.id: i for i, c in enumerate(classes)}
    class_name = {c.id: c.name for c in classes}
    task = project.task_type

    images = list(project.images)
    if approved_only:
        images = [im for im in images if im.status == "approved"]
    if not images:
        raise RuntimeError("No approved images to export." if approved_only
                           else "No images to export — upload something first.")

    out_dir = storage.project_dir(project.id) / "_raw_export"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if task != "classify":
        (out_dir / "images").mkdir()
        (out_dir / "labels").mkdir()

    for im in images:
        src = storage.image_path(project.id, im.filename)
        if not src.exists():
            continue
        anns = _ann_dicts(im)
        if task == "classify":
            tag = next((a for a in anns if a["kind"] == "classification"), None)
            cname = class_name.get(tag["class_id"], "_unlabeled") if tag else "_unlabeled"
            cdir = out_dir / cname
            cdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, cdir / im.filename)
        else:
            shutil.copy(src, out_dir / "images" / im.filename)
            _write_label_txt(out_dir / "labels" / f"{Path(im.filename).stem}.txt",
                             anns, class_idx, im.width, im.height, task)

    if task != "classify":
        (out_dir / "classes.txt").write_text("\n".join(c.name for c in classes))
        data_yaml = {"path": str(out_dir.resolve()), "train": "images", "val": "images",
                     "names": {i: c.name for i, c in enumerate(classes)}}
        (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    zip_path = out_dir.with_suffix(".zip")
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir))
    shutil.rmtree(out_dir)
    return zip_path
