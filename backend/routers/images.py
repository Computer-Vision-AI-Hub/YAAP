import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import update
from sqlalchemy.orm import Session

from ..config import ALLOWED_IMAGE_EXT
from ..db import Annotation, ImageAsset, Proposal, get_db
from ..schemas import ImagePatch
from ..services import storage
from .projects import project_or_404

router = APIRouter(prefix="/api/projects/{pid}/images", tags=["images"])


@router.post("/clear-splits")
def clear_splits(pid: int, db: Session = Depends(get_db)):
    """Forget every image's pinned split (set by an import's train/ valid/ test/ folders), so a version's
    split ratios apply to all images. Declared before the /{iid} routes."""
    project_or_404(db, pid)
    n = db.execute(update(ImageAsset).where(ImageAsset.project_id == pid, ImageAsset.split != "")
                   .values(split="").execution_options(synchronize_session=False)).rowcount
    db.commit()
    return {"cleared": n}


def serialize_image(im: ImageAsset) -> dict:
    model_conf = [a.confidence for a in im.annotations if a.source == "model"]
    return {"min_conf": min(model_conf) if model_conf else None,
            "uncertainty": im.score.uncertainty if im.score else None,
            "proposal_count": len(im.proposals),
            "id": im.id, "filename": im.filename, "width": im.width, "height": im.height,
            "status": im.status, "split": im.split,
            "annotation_count": len(im.annotations)}


@router.get("")
def list_images(pid: int, status: str = "", db: Session = Depends(get_db)):
    """Same fields as serialize_image(), but counted in three grouped queries for the whole project —
    per-image lazy loads made this take 6 s for 5 000 images (~15 000 queries)."""
    from sqlalchemy import case, func
    from ..db import ImageScore
    project_or_404(db, pid)
    q = db.query(ImageAsset).filter(ImageAsset.project_id == pid)
    if status:
        q = q.filter(ImageAsset.status == status)
    ims = q.order_by(ImageAsset.id).all()
    in_project = db.query(ImageAsset.id).filter(ImageAsset.project_id == pid)
    ann = {iid: (n, mc) for iid, n, mc in db.query(
        Annotation.image_id, func.count(Annotation.id),
        func.min(case((Annotation.source == "model", Annotation.confidence))))
        .filter(Annotation.image_id.in_(in_project)).group_by(Annotation.image_id)}
    props = dict(db.query(Proposal.image_id, func.count(Proposal.id))
                 .filter(Proposal.image_id.in_(in_project)).group_by(Proposal.image_id))
    unc = dict(db.query(ImageScore.image_id, ImageScore.uncertainty).filter(ImageScore.image_id.in_(in_project)))
    return [{"min_conf": ann.get(im.id, (0, None))[1], "uncertainty": unc.get(im.id),
             "proposal_count": props.get(im.id, 0),
             "id": im.id, "filename": im.filename, "width": im.width, "height": im.height,
             "status": im.status, "split": im.split, "annotation_count": ann.get(im.id, (0, None))[0]}
            for im in ims]


@router.post("")
# a plain def: FastAPI runs it in a worker thread — as async def, its long synchronous work froze the whole
# server (every other request waited until it finished: 12.6 s for a 500-image import)
def upload_images(pid: int, files: list[UploadFile], db: Session = Depends(get_db)):
    project_or_404(db, pid)
    created, skipped = [], []
    for f in files:
        ext = Path(f.filename or "upload").suffix.lower()
        if ext not in ALLOWED_IMAGE_EXT:
            skipped.append(f.filename)
            continue
        raw = f.file.read()
        try:
            name, w, h = storage.save_upload(pid, Path(f.filename).name, raw)
        except Exception:
            skipped.append(f.filename)
            continue
        im = ImageAsset(project_id=pid, filename=name, width=w, height=h)
        db.add(im); db.commit()
        created.append(serialize_image(im))
    return {"created": created, "skipped": skipped}


@router.get("/{iid}/proposals")
def get_proposals(pid: int, iid: int, db: Session = Depends(get_db)):
    """Class-less objects found by SAM's generic search, waiting to be labeled."""
    im = _img(db, pid, iid)
    out = []
    for pr in sorted(im.proposals, key=lambda x: -x.score):
        d = json.loads(pr.data or "{}")
        out.append({"id": pr.id, "polygon": d.get("polygon", []), "bbox": d.get("bbox", {}), "score": pr.score})
    return out


@router.delete("/{iid}/proposals")
def clear_proposals(pid: int, iid: int, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    db.query(Proposal).filter(Proposal.image_id == im.id).delete()
    db.commit()
    return {"ok": True}


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
    if body.expect_annotations is not None:
        # a no-op UPDATE first takes the write lock, so a Bootstrap round can't swap the
        # predictions between this check and the status change
        db.execute(update(ImageAsset).where(ImageAsset.id == iid).values(status=ImageAsset.status)
                   .execution_options(synchronize_session=False))
        db.expire(im)
        if sorted(a.id for a in im.annotations) != sorted(body.expect_annotations):
            db.rollback()
            raise HTTPException(409, "This image's annotations changed while you were looking — "
                                     "showing the current ones.")
    if body.status is not None:
        im.status = body.status
        if body.status == "annotated" and not im.annotations:
            im.status = "unannotated"     # "labeled" needs labels: un-approving an empty image (annotator-7)
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
