import json
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import DATA_DIR, MODELS_DIR, PRETRAINED_MODELS, RFDETR_VARIANTS, RTDETR_VARIANTS
from ..db import (Annotation, AutolabelJob, DatasetVersion, ImageAsset, LabelClass,
                  ModelWeight, SessionLocal, TrainJob, get_db)
from ..schemas import InferenceRequest, TestModelRequest
from ..services import exporter, inference, storage
from .images import serialize_image
from .projects import PALETTE, project_or_404

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def list_models(pid: int = 0, task: str = "detect", db: Session = Depends(get_db)):
    weights = db.query(ModelWeight).filter(
        (ModelWeight.project_id == None) | (ModelWeight.project_id == pid)  # noqa: E711
    ).order_by(ModelWeight.created_at.desc()).all()
    rfdetr_ok = True
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        rfdetr_ok = False
    return {
        "pretrained": PRETRAINED_MODELS.get(task, []),
        "rfdetr": RFDETR_VARIANTS if (task == "detect") else [],
        "rfdetr_available": rfdetr_ok,
        "rtdetr": RTDETR_VARIANTS if (task == "detect") else [],  # ships in ultralytics, always available
        "weights": [{"id": w.id, "name": w.name, "arch": w.arch, "task": w.task,
                     "source": w.source} for w in weights if w.task == task or not w.task],
    }


@router.post("/models/upload")
async def upload_weight(file: UploadFile, name: str = "", task: str = "detect",
                        pid: int = 0, db: Session = Depends(get_db)):
    if not (file.filename or "").endswith(".pt"):
        raise HTTPException(400, "Expected a .pt weights file")
    dest = MODELS_DIR / Path(file.filename).name
    i, stem = 1, dest.stem
    while dest.exists():
        dest = MODELS_DIR / f"{stem}_{i}.pt"; i += 1
    dest.write_bytes(await file.read())
    w = ModelWeight(project_id=pid or None, name=name or dest.stem, arch=dest.stem,
                    task=task, path=str(dest), source="uploaded")
    db.add(w); db.commit()
    return {"id": w.id, "name": w.name}


def serialize_autolabel_job(j: AutolabelJob) -> dict:
    return {"id": j.id, "project_id": j.project_id, "model_name": j.model_name,
            "status": j.status, "total": j.total, "processed": j.processed,
            "images_labeled": j.images_labeled, "annotations_added": j.annotations_added,
            "skipped": json.loads(j.skipped or "[]"), "log": j.log, "error": j.error,
            "created_at": j.created_at.isoformat()}


@router.post("/projects/{pid}/autolabel")
def autolabel(pid: int, req: InferenceRequest, background: BackgroundTasks,
             db: Session = Depends(get_db)):
    """Kick off a background auto-label run — see _run_autolabel() for the
    actual per-image loop. Returns immediately so the UI can poll progress."""
    project = project_or_404(db, pid)
    images = (db.query(ImageAsset).filter(ImageAsset.id.in_(req.image_ids),
                                          ImageAsset.project_id == pid).all()
              if req.image_ids else list(project.images))
    if not images:
        raise HTTPException(400, "No images to label")
    if req.model_name not in RFDETR_VARIANTS:   # RF-DETR never loads a file
        try:
            inference.resolve_weights(req.model_name, db)
        except inference.InferenceError as e:
            raise HTTPException(400, str(e))

    job = AutolabelJob(project_id=pid, model_name=req.model_name, total=len(images))
    db.add(job); db.commit()
    background.add_task(_run_autolabel, pid, job.id, req)
    return serialize_autolabel_job(job)


def _run_autolabel(pid: int, jid: int, req: InferenceRequest):
    """Runs after the response is sent — its own DB session, since the
    request's session is already closed by then (same pattern as
    datasets._build_version)."""
    db = SessionLocal()
    try:
        job = db.get(AutolabelJob, jid)
        project = project_or_404(db, pid)
        images = (db.query(ImageAsset).filter(ImageAsset.id.in_(req.image_ids),
                                              ImageAsset.project_id == pid).all()
                  if req.image_ids else list(project.images))
        by_name = {c.name: c for c in project.classes}
        labeled_ids, skipped, log_lines = [], [], []

        for im in images:
            path = storage.image_path(pid, im.filename)
            if not path.exists():
                skipped.append(im.filename)
                log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — file missing, skipped")
            else:
                try:
                    preds = inference.predict(req.model_name, str(path), project.task_type,
                                              req.conf, req.iou, req.device, db)
                except inference.InferenceError as e:
                    skipped.append(im.filename)
                    log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — inference failed: {e}")
                    preds = None

                if preds is not None:
                    if req.class_filter:
                        preds = [p for p in preds if p["class_name"] in req.class_filter]
                    if req.replace_existing:
                        db.query(Annotation).filter(Annotation.image_id == im.id).delete()

                    added = 0
                    for p in preds:
                        cls = by_name.get(p["class_name"])
                        if cls is None:
                            if not req.create_missing_classes:
                                continue
                            cls = LabelClass(project_id=pid, name=p["class_name"],
                                             color=PALETTE[len(by_name) % len(PALETTE)],
                                             order_idx=len(by_name))
                            db.add(cls); db.flush()
                            by_name[p["class_name"]] = cls
                        db.add(Annotation(image_id=im.id, class_id=cls.id, kind=p["kind"],
                                          data=json.dumps(p["data"]), source="model",
                                          confidence=p["confidence"]))
                        added += 1
                    if added:
                        im.status = "auto-annotated"
                        labeled_ids.append(im.id)
                        job.images_labeled += 1
                        job.annotations_added += added
                        log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — {added} object(s) found")
                    else:
                        log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — nothing detected")

            job.processed += 1
            job.skipped = json.dumps(skipped)
            job.labeled_image_ids = json.dumps(labeled_ids)
            job.log = "\n".join(log_lines)
            db.commit()

        job.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        job = db.get(AutolabelJob, jid)
        if job:
            job.status = "failed"
            job.error = str(e)
            db.commit()
    finally:
        db.close()


@router.get("/projects/{pid}/autolabel/{jid}")
def get_autolabel_job(pid: int, jid: int, db: Session = Depends(get_db)):
    job = db.get(AutolabelJob, jid)
    if not job or job.project_id != pid:
        raise HTTPException(404, "Job not found")
    return serialize_autolabel_job(job)


@router.get("/projects/{pid}/autolabel/{jid}/preview")
def autolabel_preview(pid: int, jid: int, db: Session = Depends(get_db)):
    """The images this specific job labeled, with their (now-editable)
    annotations, so the UI can render a preview grid to approve from."""
    job = db.get(AutolabelJob, jid)
    if not job or job.project_id != pid:
        raise HTTPException(404, "Job not found")
    from .annotations import serialize_ann
    from .images import serialize_image
    ids = json.loads(job.labeled_image_ids or "[]")
    images = db.query(ImageAsset).filter(ImageAsset.id.in_(ids)).all()
    images.sort(key=lambda im: ids.index(im.id))
    return {"job": serialize_autolabel_job(job),
            "images": [{**serialize_image(im), "annotations": [serialize_ann(a) for a in im.annotations]}
                       for im in images]}


@router.post("/projects/{pid}/test-model")
def test_model(pid: int, req: TestModelRequest, db: Session = Depends(get_db)):
    """Read-only: run a model over a dataset version's test-split images and
    return its raw predictions for visualization — nothing is saved."""
    project = project_or_404(db, pid)
    version = db.get(DatasetVersion, req.version_id)
    if not version or version.project_id != pid:
        raise HTTPException(404, "Version not found")
    if version.status != "ready":
        raise HTTPException(400, f"Version is '{version.status}' — wait until it is ready.")

    cfg = json.loads(version.config or "{}")
    # mirror exporter.generate()'s own inclusion filter before splitting, so this
    # recomputes the exact same test set the version was actually built from —
    # not every image in the project
    eligible = [im for im in project.images if cfg.get("include_unannotated") or im.annotations]
    if cfg.get("approved_only", True):
        eligible = [im for im in eligible if im.status == "approved"]
    test_images = sorted(exporter._split_images(eligible, cfg.get("splits", {}))["test"],
                         key=lambda im: im.id)
    if not test_images:
        raise HTTPException(400, "This version's test split is empty — regenerate it with a "
                                 "non-zero test ratio, or pin some images to the test split.")

    if req.model_name not in RFDETR_VARIANTS:   # RF-DETR never loads a file
        try:
            inference.resolve_weights(req.model_name, db)
        except inference.InferenceError as e:
            raise HTTPException(400, str(e))

    limit = max(1, min(req.limit or 20, 100))
    offset = max(0, req.offset)
    page = test_images[offset:offset + limit]

    by_name = {c.name: c for c in project.classes}
    results, skipped = [], []
    for im in page:
        path = storage.image_path(pid, im.filename)
        if not path.exists():
            skipped.append(im.filename)
            continue
        try:
            preds = inference.predict(req.model_name, str(path), project.task_type,
                                      req.conf, req.iou, req.device, db)
        except inference.InferenceError:
            skipped.append(im.filename)
            continue
        for p in preds:
            cls = by_name.get(p["class_name"])
            p["class_id"] = cls.id if cls else None
            p["color"] = cls.color if cls else "#999999"
            p["source"] = "model"
        results.append({**serialize_image(im), "predictions": preds})

    return {
        "version_id": version.id, "model_name": req.model_name,
        "test_split_total": len(test_images), "shown": len(results),
        "offset": offset, "limit": limit, "skipped": skipped, "images": results,
    }


@router.get("/system/device")
def device_info():
    return inference.torch_info()


# Host-side auto-shutdown 
LEAVE_MARKER = DATA_DIR / ".leave_at"


@router.post("/system/leaving")
def leaving(db: Session = Depends(get_db)):
    active = db.query(TrainJob).filter(TrainJob.status.in_(["queued", "running"])).first()
    if active:
        return {"ok": False, "reason": "training job active"}
    LEAVE_MARKER.write_text(str(int(time.time())))
    return {"ok": True}


@router.post("/system/here")
def here():
    LEAVE_MARKER.unlink(missing_ok=True)
    return {"ok": True}
