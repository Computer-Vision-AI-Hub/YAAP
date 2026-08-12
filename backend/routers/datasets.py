import io
import json
import zipfile

import cv2
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from ..db import DatasetVersion, ImageAsset, SessionLocal, get_db
from ..schemas import AugmentPreviewRequest, MergeRequest, VersionCreate
from ..services import augmenter, exporter, importer, storage
from .images import _img
from .projects import PALETTE, project_or_404

router = APIRouter(prefix="/api/projects/{pid}", tags=["datasets"])


@router.post("/augment/preview")
def augment_preview(pid: int, req: AugmentPreviewRequest, db: Session = Depends(get_db)):
    """Returns a zip-free multi-image preview: a single JPEG contact sheet."""
    im = _img(db, pid, req.image_id)
    img = augmenter.load_bgr(str(storage.image_path(pid, im.filename)))
    pipeline = augmenter.compose(augmenter.build_augment(req.ops), with_targets=False)
    tiles = []
    for _ in range(max(1, min(req.count, 8))):
        tiles.append(pipeline(image=img)["image"])
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT,
                                value=(16, 18, 22)) for t in tiles]
    sheet = cv2.hconcat(tiles)
    scale = min(1.0, 1600 / sheet.shape[1])
    if scale < 1.0:
        sheet = cv2.resize(sheet, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "Preview encoding failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


def serialize_version(v: DatasetVersion) -> dict:
    return {"id": v.id, "name": v.name, "status": v.status,
            "config": json.loads(v.config or "{}"), "stats": json.loads(v.stats or "{}"),
            "created_at": v.created_at.isoformat()}


@router.get("/export")
def export_raw(pid: int, approved_only: bool = False, db: Session = Depends(get_db)):
    """Pack + download the project's images/annotations as-is — no version,
    no split, no augmentation. Rebuilt fresh on every call."""
    p = project_or_404(db, pid)
    try:
        zip_path = exporter.export_raw(p, approved_only)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    fname = f"YAAP_{p.name.replace(' ', '_')}_raw.zip"
    return FileResponse(zip_path, filename=fname, media_type="application/zip")


@router.post("/import")
async def import_dataset(pid: int, file: UploadFile, default_split: str = "", db: Session = Depends(get_db)):
    """Load an external dataset zip into this project — YOLO detect/segment
    layout (data.yaml/classes.txt + images/labels), a classify folder-per-class
    layout, or a plain zip of images (imported unannotated)."""
    p = project_or_404(db, pid)
    if not (file.filename or "").endswith(".zip"):
        raise HTTPException(400, "Expected a .zip file")
    raw = await file.read()
    try:
        res = importer.import_zip(db, p, raw, PALETTE, default_split)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Not a valid zip file")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return res


@router.post("/merge")
def merge_dataset(pid: int, body: MergeRequest, db: Session = Depends(get_db)):
    """Merge another project already on this platform into this one."""
    target = project_or_404(db, pid)
    source = project_or_404(db, body.source_project_id)
    try:
        res = importer.merge_project(db, target, source, PALETTE, body.default_split)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return res


@router.get("/versions")
def list_versions(pid: int, db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    return [serialize_version(v) for v in sorted(p.versions, key=lambda v: -v.id)]


@router.post("/versions")
def create_version(pid: int, body: VersionCreate, background: BackgroundTasks,
                   db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    v = DatasetVersion(project_id=pid, name=body.name or f"v{len(p.versions) + 1}",
                       config=json.dumps(body.model_dump()), status="building")
    db.add(v); db.commit()
    background.add_task(_build_version, pid, v.id)
    return serialize_version(v)


def _build_version(pid: int, vid: int):
    db = SessionLocal()
    try:
        p = project_or_404(db, pid)
        v = db.get(DatasetVersion, vid)
        stats = exporter.generate(db, p, v)
        v.stats = json.dumps(stats)
        v.status = "ready"
    except Exception as e:
        v = db.get(DatasetVersion, vid)
        if v:
            v.status = "failed"
            v.stats = json.dumps({"error": str(e)})
    finally:
        db.commit()
        db.close()


@router.get("/versions/{vid}/download")
def download_version(pid: int, vid: int, db: Session = Depends(get_db)):
    v = db.get(DatasetVersion, vid)
    if not v or v.project_id != pid:
        raise HTTPException(404, "Version not found")
    zip_path = exporter.version_dir(pid, vid).with_suffix(".zip")
    if not zip_path.exists():
        raise HTTPException(404, "Archive not built yet")
    p = project_or_404(db, pid)
    fname = f"YAAP_{p.name.replace(' ', '_')}_{v.name}.zip"
    return FileResponse(zip_path, filename=fname, media_type="application/zip")


@router.delete("/versions/{vid}")
def delete_version(pid: int, vid: int, db: Session = Depends(get_db)):
    import shutil
    v = db.get(DatasetVersion, vid)
    if not v or v.project_id != pid:
        raise HTTPException(404, "Version not found")
    shutil.rmtree(exporter.version_dir(pid, vid), ignore_errors=True)
    exporter.version_dir(pid, vid).with_suffix(".zip").unlink(missing_ok=True)
    db.delete(v); db.commit()
    return {"ok": True}
