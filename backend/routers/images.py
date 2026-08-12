from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import ALLOWED_IMAGE_EXT
from ..db import Annotation, ImageAsset, get_db
from ..schemas import ImagePatch
from ..services import storage
from .projects import project_or_404

router = APIRouter(prefix="/api/projects/{pid}/images", tags=["images"])


def serialize_image(im: ImageAsset) -> dict:
    return {"id": im.id, "filename": im.filename, "width": im.width, "height": im.height,
            "status": im.status, "split": im.split,
            "annotation_count": len(im.annotations)}


@router.get("")
def list_images(pid: int, status: str = "", db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    ims = p.images
    if status:
        ims = [im for im in ims if im.status == status]
    return [serialize_image(im) for im in ims]


@router.post("")
async def upload_images(pid: int, files: list[UploadFile], db: Session = Depends(get_db)):
    project_or_404(db, pid)
    created, skipped = [], []
    for f in files:
        ext = Path(f.filename or "upload").suffix.lower()
        if ext not in ALLOWED_IMAGE_EXT:
            skipped.append(f.filename)
            continue
        raw = await f.read()
        try:
            name, w, h = storage.save_upload(pid, Path(f.filename).name, raw)
        except Exception:
            skipped.append(f.filename)
            continue
        im = ImageAsset(project_id=pid, filename=name, width=w, height=h)
        db.add(im); db.commit()
        created.append(serialize_image(im))
    return {"created": created, "skipped": skipped}


@router.get("/{iid}/file")
def image_file(pid: int, iid: int, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    path = storage.image_path(pid, im.filename)
    if not path.exists():
        raise HTTPException(404, "File missing on disk")
    return FileResponse(path)


@router.get("/{iid}/thumb")
def image_thumb(pid: int, iid: int, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    t = storage.thumb_path(pid, im.filename)
    return FileResponse(t if t.exists() else storage.image_path(pid, im.filename))


@router.patch("/{iid}")
def patch_image(pid: int, iid: int, body: ImagePatch, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    if body.status is not None:
        im.status = body.status
    if body.split is not None:
        im.split = body.split
    db.commit()
    return serialize_image(im)


@router.delete("/{iid}")
def delete_image(pid: int, iid: int, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    storage.image_path(pid, im.filename).unlink(missing_ok=True)
    storage.thumb_path(pid, im.filename).unlink(missing_ok=True)
    db.delete(im); db.commit()
    return {"ok": True}


def _img(db: Session, pid: int, iid: int) -> ImageAsset:
    im = db.get(ImageAsset, iid)
    if not im or im.project_id != pid:
        raise HTTPException(404, "Image not found")
    return im
