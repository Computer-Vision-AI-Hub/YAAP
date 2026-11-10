"""Project storage layout + thumbnail generation.

data/projects/<pid>/
    images/       original uploads
    thumbs/       320px previews
    versions/     generated dataset versions (v<id>/ ... + .zip)
"""
import io
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


# what the editor's <img> can show; anything else (TIFF, …) is stored as PNG
BROWSER_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
EXIF_ORIENTATION = 0x0112


def _to_8bit(im: Image.Image) -> Image.Image:
    """16-bit / float images (common in TIFF) → 8-bit, stretched over their own value range."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        lo, hi = im.getextrema()
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        return im.point(lambda v: (v - lo) * scale).convert("L")
    if im.mode not in ("1", "L", "LA", "RGB", "RGBA"):
        return im.convert("RGBA" if "A" in im.getbands() else "RGB")
    return im


def save_upload(pid: int, filename: str, raw: bytes) -> tuple[str, int, int]:
    """Persist an upload, de-duplicate names, build a thumbnail. Returns (name, w, h).
    The bytes are checked before anything is written (a corrupt file used to stay on disk and push the
    next upload to name_1). Rotated photos (EXIF orientation) are stored upright, and formats a browser
    can't display (TIFF) as PNG, so every later step sees exactly the pixels the editor shows."""
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.load()
    except Exception as e:
        raise ValueError(f"not a readable image: {e}") from e

    d = project_dir(pid) / "images"
    name = Path(filename)
    convert = name.suffix.lower() not in BROWSER_EXT
    if convert:
        name = name.with_suffix(".png")
    target = d / name.name
    stem, suffix = target.stem, target.suffix
    i = 1
    while target.exists():
        target = d / f"{stem}_{i}{suffix}"
        i += 1

    with Image.open(io.BytesIO(raw)) as im:
        rotated = im.getexif().get(EXIF_ORIENTATION, 1) not in (0, 1)
        up = ImageOps.exif_transpose(im)      # auto-orient once, at ingest
        if convert:
            _to_8bit(up).save(target, "PNG")
        elif rotated:                         # re-encode upright, without the orientation tag
            fmt = (im.format or "JPEG").upper()
            if fmt in ("JPEG", "MPO"):
                up.convert("RGB").save(target, "JPEG", quality=95)
            else:
                up.save(target, fmt)
        else:
            target.write_bytes(raw)           # untouched: no re-encoding of ordinary uploads
        w, h = up.size
        thumb = _to_8bit(up).convert("RGB")
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(thumb_path(pid, target.name), "JPEG", quality=85)
    return target.name, w, h
