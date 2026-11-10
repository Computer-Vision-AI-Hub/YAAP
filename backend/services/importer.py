"""Bring external data into a project — either an uploaded dataset zip
(YOLO detect/segment layout with data.yaml, or a classify folder-per-class
layout), or another project already on this platform (merge).

"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import yaml

from ..db import Annotation, ImageAsset, LabelClass, Project, next_class_index
from . import storage

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
# Roboflow exports use "valid/" for the validation split.
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def _get_or_create_class(db, project: Project, by_name: dict, name: str, palette: list[str]) -> LabelClass:
    cls = by_name.get(name)
    if cls is None:
        cls = LabelClass(project_id=project.id, name=name,
                         color=palette[len(by_name) % len(palette)],
                         order_idx=next_class_index(db, project.id))
        db.add(cls); db.flush()
        by_name[name] = cls
    return cls


def _read_class_manifest(zf: zipfile.ZipFile, names: list[str]) -> list[str] | None:
    yaml_entry = next((n for n in names if Path(n).name == "data.yaml"), None)
    if yaml_entry:
        cfg = yaml.safe_load(zf.read(yaml_entry).decode("utf-8", "ignore")) or {}
        nm = cfg.get("names")
        if isinstance(nm, dict):
            return [nm[k] for k in sorted(nm, key=lambda x: int(x))]
        if isinstance(nm, list):
            return list(nm)
    classes_entry = next((n for n in names if Path(n).name in ("classes.txt", "labels.txt")), None)
    if classes_entry:
        return [l.strip() for l in zf.read(classes_entry).decode("utf-8", "ignore").splitlines() if l.strip()]
    return None


def import_zip(db, project: Project, raw: bytes, palette: list[str], default_split: str = "") -> dict:
    """Detect + supports/hits:
      detect/segment — <anything>/images/*.jpg + matching <anything>/labels/*.txt,
                       or a flat images/*.jpg + labels/*.txt, class names from
                       data.yaml or classes.txt (falls back to "class<i>").
      classify       — <split?>/<class_name>/*.jpg
      plain images   — any zip of just images, imported unannotated.
    """
    zf = zipfile.ZipFile(raw if hasattr(raw, "read") else io.BytesIO(raw))     # a file object or bytes
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        raise ValueError("Zip is empty.")

    by_name = {c.name: c for c in project.classes}
    class_names = _read_class_manifest(zf, names)
    if class_names:
        for nm in class_names:
            _get_or_create_class(db, project, by_name, nm, palette)

    task = project.task_type
    img_entries = [n for n in names if Path(n).suffix.lower() in IMG_EXT]
    if not img_entries:
        raise ValueError("No images found in the zip.")
    label_files = {n for n in names
                   if Path(n).suffix.lower() == ".txt" and Path(n).name not in ("classes.txt", "labels.txt")}
    labels_by_stem: dict[str, list[str]] = {}
    for n in label_files:
        labels_by_stem.setdefault(Path(n).stem, []).append(n)

    created, skipped = 0, 0
    bad_lines: list[str] = []
    for n in img_entries:
        parts_lower = [p.lower() for p in Path(n).parts]
        split = next((SPLIT_ALIASES[p] for p in parts_lower if p in SPLIT_ALIASES), default_split)
        try:
            fname, w, h = storage.save_upload(project.id, Path(n).name, zf.read(n))
        except Exception:
            skipped += 1
            continue
        im = ImageAsset(project_id=project.id, filename=fname, width=w, height=h, split=split)
        db.add(im); db.flush()

        if task == "classify":
            cname = Path(n).parent.name
            cls = _get_or_create_class(db, project, by_name, cname, palette)
            db.add(Annotation(image_id=im.id, class_id=cls.id, kind="classification", data="{}"))
            im.status = "annotated"
        else:
            lbl = _label_for(n, label_files, labels_by_stem)
            n_anns = 0
            if lbl:
                for ln, line in enumerate(zf.read(lbl).decode("utf-8", "ignore").splitlines(), 1):
                    fields = line.split()
                    if not fields:
                        continue
                    try:                  # a malformed line is skipped (and reported), not fatal
                        if len(fields) < 5:
                            raise ValueError("fewer than 5 values")
                        ci = int(fields[0])
                        vals = list(map(float, fields[1:]))
                        if ci < 0:
                            raise ValueError("negative class index")
                    except ValueError as e:
                        bad_lines.append(f"{lbl}:{ln}: {e}")
                        continue
                    cname = class_names[ci] if class_names and ci < len(class_names) else f"class{ci}"
                    cls = _get_or_create_class(db, project, by_name, cname, palette)
                    # Boxes and polygons are converted to whichever one the
                    # project's task uses (e.g. a segmentation export into a
                    # detect project keeps each polygon's bounding box).
                    if len(vals) == 4:
                        cx, cy, bw, bh = vals
                        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    elif len(vals) >= 6 and len(vals) % 2 == 0:
                        pts = [[vals[i] * w, vals[i + 1] * h] for i in range(0, len(vals), 2)]
                    else:
                        continue
                    if task == "detect":
                        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                        data = {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}
                        kind = "bbox"
                    else:
                        data = {"points": pts}
                        kind = "polygon"
                    db.add(Annotation(image_id=im.id, class_id=cls.id, kind=kind, data=json.dumps(data)))
                    n_anns += 1
            im.status = "annotated" if n_anns else "unannotated"
        created += 1
        db.commit()

    return {"images_created": created, "images_skipped": skipped, "classes": list(by_name.keys()),
            "label_lines_skipped": len(bad_lines), "label_line_errors": bad_lines[:20]}


def _label_for(img: str, label_files: set[str], labels_by_stem: dict[str, list[str]]) -> str | None:
    """The label file of an image entry: the same path with images/ → labels/ first, then a .txt next to
    the image, then a file with the same stem (if several splits have one, the one sharing the most
    folders with the image). Matching by stem alone gave train/x.jpg the labels of valid/x.txt."""
    p = Path(img)
    parts = list(p.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt").as_posix()
            if cand in label_files:
                return cand
            break
    cand = p.with_suffix(".txt").as_posix()
    if cand in label_files:
        return cand
    options = labels_by_stem.get(p.stem, [])
    if len(options) <= 1:
        return options[0] if options else None

    def shared(lbl: str) -> int:
        n = 0
        for a, b in zip(Path(lbl).parts, parts):
            if a != b:
                break
            n += 1
        return n
    return max(options, key=shared)


def merge_project(db, target: Project, source: Project, palette: list[str], default_split: str = "") -> dict:
    """Copy every image (+ its annotations, classes matched by name) from
    `source` into `target`. Both projects must share a task type."""
    if source.id == target.id:
        raise ValueError("Cannot merge a project into itself.")
    if source.task_type != target.task_type:
        raise ValueError(f"Task type mismatch — source is '{source.task_type}', target is '{target.task_type}'.")

    by_name = {c.name: c for c in target.classes}
    src_class_name = {c.id: c.name for c in source.classes}

    created, skipped = 0, 0
    for im in list(source.images):
        src_path = storage.image_path(source.id, im.filename)
        if not src_path.exists():
            skipped += 1
            continue
        try:
            fname, w, h = storage.save_upload(target.id, im.filename, src_path.read_bytes())
        except Exception:
            skipped += 1
            continue
        new_im = ImageAsset(project_id=target.id, filename=fname, width=w, height=h,
                            split=im.split or default_split, status=im.status)
        db.add(new_im); db.flush()
        for a in im.annotations:
            cname = src_class_name.get(a.class_id)
            if not cname:
                continue
            cls = _get_or_create_class(db, target, by_name, cname, palette)
            db.add(Annotation(image_id=new_im.id, class_id=cls.id, kind=a.kind,
                              data=a.data, source=a.source, confidence=a.confidence, origin=a.origin or ""))
        created += 1
        db.commit()

    return {"images_merged": created, "images_skipped": skipped}
