import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import AutolabelJob, BootstrapRun, ModelWeight, get_db
from ..schemas import BootstrapRequest
from ..services import bootstrap
from .models import serialize_autolabel_job
from .projects import project_or_404

router = APIRouter(prefix="/api/projects/{pid}/bootstrap", tags=["bootstrap"])


def serialize_run(db: Session, r: BootstrapRun) -> dict:
    aj = db.get(AutolabelJob, r.autolabel_job_id) if r.autolabel_job_id else None
    w = db.get(ModelWeight, r.weight_id) if r.weight_id else None
    return {"id": r.id, "round": r.round, "status": r.status, "message": r.message, "error": r.error,
            "model_arch": r.model_arch, "epochs": r.epochs, "epoch": r.epoch,
            "train_images": r.train_images, "target_images": r.target_images,
            "train_job_id": r.train_job_id, "weight_id": r.weight_id, "weight_name": w.name if w else None,
            "autolabel": serialize_autolabel_job(aj) if aj else None,
            "labeled_image_ids": json.loads(aj.labeled_image_ids or "[]") if aj else [],
            "to_review": bootstrap.to_review(db, r),
            "created_at": r.created_at.isoformat(),
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}


@router.get("")
def bootstrap_state(pid: int, db: Session = Depends(get_db)):
    s = bootstrap.summary(db, project_or_404(db, pid))
    last = s.pop("last_run")
    return {**s, "last": serialize_run(db, last) if last else None}


@router.post("")
def start_bootstrap(pid: int, req: BootstrapRequest, db: Session = Depends(get_db)):
    """Train a quick model on the labeled images, then label the rest with it —
    returns at once; poll GET /bootstrap/{bid} for progress."""
    p = project_or_404(db, pid)
    try:
        run = bootstrap.start(db, p, req)
    except bootstrap.BootstrapBusy as e:
        raise HTTPException(409, str(e))
    except bootstrap.BootstrapError as e:
        raise HTTPException(400, str(e))
    return serialize_run(db, run)


@router.get("/{bid}")
def get_bootstrap(pid: int, bid: int, db: Session = Depends(get_db)):
    return serialize_run(db, bootstrap.refresh(db, _run(db, pid, bid)))


@router.post("/{bid}/stop")
def stop_bootstrap(pid: int, bid: int, db: Session = Depends(get_db)):
    return serialize_run(db, bootstrap.stop(db, _run(db, pid, bid)))


def _run(db: Session, pid: int, bid: int) -> BootstrapRun:
    r = db.get(BootstrapRun, bid)
    if not r or r.project_id != pid:
        raise HTTPException(404, "Bootstrap run not found")
    return r
