import datetime as dt
import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import MODELS_DIR, PRETRAINED_MODELS, RFDETR_VARIANTS, RTDETR_VARIANTS
from ..db import (Annotation, AutolabelJob, DatasetVersion, ImageAsset, ImageScore, LabelClass,
                  ModelWeight, SessionLocal, get_db, next_class_index)
from ..schemas import InferenceRequest, TestModelRequest
from ..services import exporter, inference, jobs, storage
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
# a plain def: FastAPI runs it in a worker thread — as async def, its long synchronous work froze the whole
# server (every other request waited until it finished: 12.6 s for a 500-image import)
def upload_weight(file: UploadFile, name: str = "", task: str = "detect",
                        pid: int = 0, db: Session = Depends(get_db)):
    if not (file.filename or "").endswith(".pt"):
        raise HTTPException(400, "Expected a .pt weights file")
    dest = MODELS_DIR / Path(file.filename).name
    i, stem = 1, dest.stem
    while dest.exists():
        dest = MODELS_DIR / f"{stem}_{i}.pt"; i += 1
    dest.write_bytes(file.file.read())
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
        names = {c.id: c.name for c in project.classes}
        labeled_ids, skipped, log_lines = [], [], []
        seen = {im.id: im.status for im in images}   # what the job saw; a change means a human edited it

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
                    preds = [p for p in preds if p["class_name"] in by_name or req.create_missing_classes]

                if preds is not None and not preds:
                    # nothing to replace them with: existing annotations (and status) stay as they are
                    log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — nothing detected"
                                     + (", existing annotations kept" if req.replace_existing and im.annotations else ""))
                elif preds is not None and not jobs.claim_image(db, im.id, seen[im.id]):
                    db.rollback()
                    log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — "
                                     "changed by hand meanwhile, left alone")
                elif preds is not None:
                    db.expire(im, ["annotations"])         # re-read under the write lock
                    existing = [] if req.replace_existing else \
                        [jobs.shape(names.get(a.class_id), a.kind, a.data_obj) for a in im.annotations]
                    # one box per object — a model torn between two classes, or a box already there
                    kept, dropped = jobs.drop_duplicates(
                        [jobs.shape(p["class_name"], p["kind"], p["data"], conf=p["confidence"], p=p) for p in preds],
                        existing, req.max_overlap)
                    note = f", {dropped} duplicate(s) skipped" if dropped else ""
                    if not kept:                           # every detection sits on an object already labeled
                        db.rollback()
                        log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — nothing new{note}")
                    else:
                        if req.replace_existing:
                            db.query(Annotation).filter(Annotation.image_id == im.id).delete()
                        added = 0
                        for k in kept:
                            p = k["p"]
                            cls = by_name.get(p["class_name"])
                            if cls is None:                # create_missing_classes (the only way here)
                                cls = LabelClass(project_id=pid, name=p["class_name"],
                                                 color=PALETTE[len(by_name) % len(PALETTE)],
                                                 order_idx=next_class_index(db, pid))
                                db.add(cls); db.flush()
                                by_name[cls.name], names[cls.id] = cls, cls.name
                            db.add(Annotation(image_id=im.id, class_id=cls.id, kind=p["kind"],
                                              data=json.dumps(p["data"]), source="model",
                                              confidence=p["confidence"], origin="autolabel"))
                            added += 1
                        im.status = "auto-annotated"
                        labeled_ids.append(im.id)
                        job.images_labeled += 1
                        job.annotations_added += added
                        log_lines.append(f"[{job.processed + 1}/{job.total}] {im.filename} — {added} object(s) found{note}")

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
        jobs.job_finished()


UNCHECKED_STATUSES = ("unannotated", "auto-annotated", "review")


class ScoreRequest(BaseModel):
    model_name: str
    device: str = "auto"
    only_unannotated: bool = True


@router.post("/projects/{pid}/active-learning/score")
def score_uncertainty(pid: int, req: ScoreRequest, background: BackgroundTasks,
                      db: Session = Depends(get_db)):
    """Active learning: run a model over (unannotated) images and store how
    uncertain it is about each — the Annotate view can then show the most
    informative images first. Writes no annotations."""
    p = project_or_404(db, pid)
    if req.model_name not in RFDETR_VARIANTS:
        try:
            inference.resolve_weights(req.model_name, db)
        except inference.InferenceError as e:
            raise HTTPException(400, str(e))
    # "unannotated" = nobody has checked it yet: no labels, or only unreviewed model predictions
    # (after a Bootstrap round / find-all almost every image holds some — those are worth ranking too)
    images = [im for im in p.images if not req.only_unannotated or im.status in UNCHECKED_STATUSES]
    if not images:
        raise HTTPException(400, "No images to score.")
    job = AutolabelJob(project_id=pid, model_name=f"uncertainty · {req.model_name}",
                       total=len(images), status="running")
    db.add(job); db.commit()
    background.add_task(_run_scoring, pid, job.id, [im.id for im in images], req)
    return serialize_autolabel_job(job)


def _run_scoring(pid: int, jid: int, image_ids: list[int], req: "ScoreRequest"):
    db = SessionLocal()
    try:
        job = db.get(AutolabelJob, jid)
        task = project_or_404(db, pid).task_type
        skipped, log = [], []
        for n, iid in enumerate(image_ids, 1):
            im = db.get(ImageAsset, iid)
            if im is None:
                job.processed += 1
                continue
            path = storage.image_path(pid, im.filename)
            try:
                if not path.exists():
                    raise inference.InferenceError("file missing")
                u = inference.uncertainty(req.model_name, str(path), task, req.device, db)
                row = im.score or ImageScore(image_id=im.id)
                row.uncertainty, row.model_name = u, req.model_name
                row.created_at = dt.datetime.utcnow()
                db.add(row)
                job.images_labeled += 1
                log.append(f"[{n}/{job.total}] {im.filename} — uncertainty {u:.2f}")
            except inference.InferenceError as e:
                skipped.append(im.filename)
                log.append(f"[{n}/{job.total}] {im.filename} — failed: {e}")
            job.processed += 1
            job.skipped = json.dumps(skipped)
            job.log = "\n".join(log[-200:])
            db.commit()
        job.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        job = db.get(AutolabelJob, jid)
        if job:
            job.status, job.error = "failed", str(e)
            db.commit()
    finally:
        db.close()
        jobs.job_finished()


@router.get("/projects/{pid}/active-learning")
def active_learning_state(pid: int, db: Session = Depends(get_db)):
    project_or_404(db, pid)
    rows = (db.query(ImageScore).join(ImageAsset, ImageAsset.id == ImageScore.image_id)
            .filter(ImageAsset.project_id == pid).order_by(ImageScore.created_at.desc()).all())
    return {"scored": len(rows), "model_name": rows[0].model_name if rows else "",
            "scored_at": rows[0].created_at.isoformat() if rows else None}


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

    # exactly the images the version put in its test split (re-deriving the split from the project's
    # current images showed training images as "test" as soon as anything had changed)
    test_ids = set(exporter.split_image_ids(version, "test"))
    test_images = sorted((im for im in project.images if im.id in test_ids), key=lambda im: im.id)
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
LEAVE_MARKER = jobs.LEAVE_MARKER


@router.post("/system/leaving")
def leaving(tab: str = "", db: Session = Depends(get_db)):
    """Tab closed. Ignored while another tab is open; while a job runs the shutdown is deferred until
    it ends (jobs.job_finished)."""
    return jobs.request_leave(db, tab or None)


@router.post("/system/here")
def here(tab: str = ""):
    """A tab is open (sent on load and as a ~20 s heartbeat)."""
    jobs.cancel_leave(tab or None)
    return {"ok": True}
