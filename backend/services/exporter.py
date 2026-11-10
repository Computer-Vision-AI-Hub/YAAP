"""Dataset version generation.

"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import yaml

from ..db import Annotation, DatasetVersion, ImageAsset, LabelClass, Project
from . import augmenter, storage


def _valid_geometry(kind: str, d: dict) -> bool:
    """A shape every writer can handle (old rows may predate the API's validation)."""
    try:
        if kind == "bbox":
            return float(d["w"]) > 0 and float(d["h"]) > 0 and float(d["x"]) == float(d["x"]) \
                and float(d["y"]) == float(d["y"])
        if kind == "polygon":
            pts = d.get("points") or []
            return len(pts) >= 3 and all(len(q) >= 2 and float(q[0]) == float(q[0]) for q in pts)
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _ann_dicts(image: ImageAsset, bad: list | None = None) -> list[dict]:
    """The image's annotations as plain dicts; broken shapes are left out (and listed in `bad`)."""
    out = []
    for a in image.annotations:
        d = a.data_obj
        if not _valid_geometry(a.kind, d):
            if bad is not None:
                bad.append(f"{image.filename}: {a.kind} {json.dumps(d)[:80]}")
            continue
        out.append({"kind": a.kind, "class_id": a.class_id, "data": d,
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
    """Roboflow-style layout: one `_annotations.coco.json` per split"""
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
    skipped: list[str] = []          # images (or augmented copies) that could not be written, and why
    written_ids: dict[str, list[int]] = {s: [] for s in split_map}   # which image went to which split
    bad_anns: list[str] = []         # broken shapes left out
    tiny_dropped, tiny_images = 0, []  # labels too small to survive the resize (reported, not an error)
    augmenter.seed_everything(1337)
    # folder per split: COCO-format versions use Roboflow's names (train/ valid/ test/ — RF-DETR and other COCO
    # tools read validation from valid/_annotations.coco.json); YOLO-only versions keep ultralytics' val/
    folder = {"train": "train", "val": "valid" if fmt == "coco" else "val", "test": "test"}

    for split, ims in split_map.items():
        if not ims:
            continue
        sdir = out_dir / folder[split]
        if task == "classify":
            (sdir).mkdir(parents=True, exist_ok=True)
        else:
            (sdir / "images").mkdir(parents=True, exist_ok=True)
            (sdir / "labels").mkdir(parents=True, exist_ok=True)

        for im in ims:
            src = storage.image_path(project.id, im.filename)
            if not src.exists():
                continue
            img = augmenter.load_bgr(str(src))
            anns = _ann_dicts(im, bad_anns)
            variants = [(pre, 0)]
            if aug is not None and split == "train":     # augment training split only
                variants += [(aug, k + 1) for k in range(multiplier - 1)]

            for pipeline, k in variants:
                try:
                    out_img, out_anns = augmenter.apply(pipeline, img, anns)
                except Exception as e:      # never silently: the version reports what is missing
                    skipped.append(f"{im.filename}{f' (augmented copy {k})' if k else ''}: {e}"[:300])
                    continue
                if k == 0 and task != "classify" and len(out_anns) < len(anns):
                    # e.g. 4-px objects in a 2000-px photo resized to 640: under albumentations' 4 px² minimum
                    tiny_dropped += len(anns) - len(out_anns)
                    if len(tiny_images) < 20:
                        tiny_images.append(im.filename)
                stem = f"{Path(im.filename).stem}_{im.id}" + (f"_aug{k}" if k else "")
                if task == "classify":
                    tag = next((a for a in out_anns if a["kind"] == "classification"), None)
                    cname = class_name.get(tag["class_id"], "_unlabeled") if tag else "_unlabeled"
                    cdir = sdir / cname
                    cdir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cdir / f"{stem}.jpg"), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    ann_count += 1 if tag else 0
                else:
                    oh, ow = out_img.shape[:2]
                    cv2.imwrite(str(sdir / "images" / f"{stem}.jpg"), out_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    _write_label_txt(sdir / "labels" / f"{stem}.txt",
                                     out_anns, class_idx, ow, oh, task)
                    ann_count += len(out_anns)
                    if fmt == "coco":                     # parallel Roboflow-style layout
                        cv2.imwrite(str(sdir / f"{stem}.jpg"), out_img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        coco_entries[split].append((stem, ow, oh, out_anns))
                counts[split] += 1
                if k == 0:
                    written_ids[split].append(im.id)

    if task != "classify":
        # no `path:` — ultralytics then resolves the splits next to data.yaml, so the downloaded zip
        # trains as-is anywhere (an absolute /app/data/... path only exists inside YAAP's container)
        data_yaml = {
            "train": "train/images", "val": f"{folder['val']}/images",
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
                _write_coco_json(out_dir / folder[split] / "_annotations.coco.json", entries, classes, class_idx)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir.parent))

    return {"images": counts, "source_images": source_counts,
            "source_images_total": sum(source_counts.values()), "format": fmt,
            "annotations": ann_count, "classes": [c.name for c in classes], "path": str(out_dir),
            "skipped": skipped, "bad_annotations": bad_anns, "split_ids": written_ids,
            "tiny_dropped": tiny_dropped, "tiny_dropped_images": tiny_images}


def version_dir(project_id: int, version_id: int) -> Path:
    return storage.project_dir(project_id) / "versions" / f"v{version_id}"


def split_image_ids(version: DatasetVersion, split: str = "test") -> list[int]:
    """Image ids a version put in `split`, as built — not re-derived from the project's current images
    (any approval/upload since then reshuffles a fresh split). New versions store them in stats; older
    ones are read back from the exported file names (<stem>_<image id>[_aug<k>].jpg)."""
    import re
    stats = json.loads(version.stats or "{}")
    if isinstance(stats.get("split_ids"), dict):
        return list(stats["split_ids"].get(split, []))
    ids: set[int] = set()
    d = version_dir(version.project_id, version.id) / split
    if split == "val" and not d.exists():
        d = d.parent / "valid"
    for f in d.rglob("*") if d.exists() else []:
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "labels" not in f.parts:
            m = re.search(r"_(\d+)(?:_aug\d+)?$", f.stem)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def export_raw(project: Project, approved_only: bool = False) -> Path:
    """Pack the project's current images + annotations as-is. Every call works in its own temporary
    folder (two downloads at once used to delete each other's files); the caller deletes the returned
    zip's folder after sending it (see remove_export)."""
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

    work = Path(tempfile.mkdtemp(prefix="_raw_export_", dir=storage.project_dir(project.id)))
    out_dir = work / "pack"
    out_dir.mkdir()
    used_stems: set[str] = set()     # same.jpg + same.png must not share labels/same.txt

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
            stem, ext = Path(im.filename).stem, Path(im.filename).suffix
            if stem in used_stems:
                stem = f"{stem}_{im.id}"
            used_stems.add(stem)
            shutil.copy(src, out_dir / "images" / f"{stem}{ext}")
            _write_label_txt(out_dir / "labels" / f"{stem}.txt",
                             anns, class_idx, im.width, im.height, task)

    # the class list (its order is the export index) — also for classification, so a re-import keeps the order
    (out_dir / "classes.txt").write_text("\n".join(c.name for c in classes))
    if task != "classify":
        data_yaml = {"train": "images", "val": "images",       # relative to data.yaml, see generate()
                     "names": {i: c.name for i, c in enumerate(classes)}}
        (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    zip_path = work / "export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir))
    shutil.rmtree(out_dir)
    return zip_path


def remove_export(zip_path: Path):
    """Delete a raw export's temporary folder (after the response has been sent)."""
    shutil.rmtree(Path(zip_path).parent, ignore_errors=True)
