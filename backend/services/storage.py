"""Project storage layout + thumbnail generation.

data/projects/<pid>/
    images/       original uploads
    thumbs/       320px previews
    versions/     generated dataset versions (v<id>/ ... + .zip)
"""
from pathlib import Path

from PIL import Image, ImageOps

from ..config import PROJECTS_DIR, THUMB_SIZE


def project_dir(pid: int) -> Path:
    d = PROJECTS_DIR / str(pid)
    for sub in ("images", "thumbs", "versions"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def image_path(pid: int, filename: str) -> Path:
    return project_dir(pid) / "images" / filename


def thumb_path(pid: int, filename: str) -> Path:
    return project_dir(pid) / "thumbs" / (Path(filename).stem + ".jpg")


def save_upload(pid: int, filename: str, raw: bytes) -> tuple[str, int, int]:
    """Persist an upload, de-duplicate names, build a thumbnail. Returns (name, w, h)."""
    d = project_dir(pid) / "images"
    target = d / filename
    stem, suffix = target.stem, target.suffix
    i = 1
    while target.exists():
        target = d / f"{stem}_{i}{suffix}"
        i += 1
    target.write_bytes(raw)

    with Image.open(target) as im:
        im = ImageOps.exif_transpose(im)      # auto-orient once, at ingest
        w, h = im.size
        thumb = im.convert("RGB").copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(thumb_path(pid, target.name), "JPEG", quality=85)
    return target.name, w, h
