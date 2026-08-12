"""Bring external data into a project — either an uploaded dataset zip
(YOLO detect/segment layout with data.yaml, or a classify folder-per-class
layout), or another project already on this platform (merge).

Matching is always by class *name*: a class already present in the target
project is reused, anything new is created (same pattern as autolabel's
create_missing_classes). Images are copied through storage.save_upload, which
de-duplicates filenames and rebuilds thumbnails, so nothing here touches the
source files.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import yaml

from ..db import Annotation, ImageAsset, LabelClass, Project
from . import storage

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def _get_or_create_class(db, project: Project, by_name: dict, name: str, palette: list[str]) -> LabelClass:
    cls = by_name.get(name)
    if cls is None:
        cls = LabelClass(project_id=project.id, name=name,
                         color=palette[len(by_name) % len(palette)], order_idx=len(by_name))
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
    zf = zipfile.ZipFile(io.BytesIO(raw))
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
    label_by_stem = {Path(n).stem: n for n in names
                     if Path(n).suffix.lower() == ".txt" and Path(n).name not in ("classes.txt", "labels.txt")}

    created, skipped = 0, 0
    for n in img_entries:
        parts_lower = [p.lower() for p in Path(n).parts]
        split = next((s for s in SPLITS if s in parts_lower), default_split)
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
            lbl = label_by_stem.get(Path(n).stem)
            n_anns = 0
            if lbl:
                for line in zf.read(lbl).decode("utf-8", "ignore").splitlines():
                    fields = line.split()
                    if len(fields) < 5:
                        continue
                    ci = int(fields[0])
                    cname = class_names[ci] if class_names and ci < len(class_names) else f"class{ci}"
                    cls = _get_or_create_class(db, project, by_name, cname, palette)
                    vals = list(map(float, fields[1:]))
                    if task == "detect" and len(vals) == 4:
                        cx, cy, bw, bh = vals
                        data = {"x": (cx - bw / 2) * w, "y": (cy - bh / 2) * h, "w": bw * w, "h": bh * h}
                        kind = "bbox"
                    elif task == "segment" and len(vals) >= 6 and len(vals) % 2 == 0:
                        pts = [[vals[i] * w, vals[i + 1] * h] for i in range(0, len(vals), 2)]
                        data = {"points": pts}
                        kind = "polygon"
                    else:
                        continue
                    db.add(Annotation(image_id=im.id, class_id=cls.id, kind=kind, data=json.dumps(data)))
                    n_anns += 1
            im.status = "annotated" if n_anns else "unannotated"
        created += 1
        db.commit()

    return {"images_created": created, "images_skipped": skipped, "classes": list(by_name.keys())}


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
                              data=a.data, source=a.source, confidence=a.confidence))
        created += 1
        db.commit()

    return {"images_merged": created, "images_skipped": skipped}
