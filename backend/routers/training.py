from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import RUNS_DIR
from ..db import DatasetVersion, TrainJob, get_db
from ..schemas import TrainRequest
from ..services import trainer
from .projects import project_or_404

router = APIRouter(prefix="/api/projects/{pid}/train", tags=["training"])

# ultralytics writes these straight into the run dir (RUNS_DIR/job_<id>/) once
# training finishes — see train_worker.py's project=/name= args
PLOT_FILES = {"results": "results.png", "confusion_matrix": "confusion_matrix.png"}


def serialize_job(j: TrainJob) -> dict:
    return {"id": j.id, "version_id": j.version_id, "model_arch": j.model_arch,
            "status": j.status, "weights_path": j.weights_path,
            "created_at": j.created_at.isoformat()}


@router.get("")
def list_jobs(pid: int, db: Session = Depends(get_db)):
    jobs = db.query(TrainJob).filter_by(project_id=pid).order_by(TrainJob.id.desc()).all()
    return [serialize_job(trainer.refresh_status(db, j)) for j in jobs]


@router.post("")
def start_training(pid: int, req: TrainRequest, db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    v = db.get(DatasetVersion, req.version_id)
    if not v or v.project_id != pid:
        raise HTTPException(404, "Version not found")
    if v.status != "ready":
        raise HTTPException(400, f"Version is '{v.status}' — wait until it is ready.")
    try:
        job = trainer.start(db, p, v, req)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return serialize_job(job)


@router.get("/{jid}/log")
def job_log(pid: int, jid: int, lines: int = 100, db: Session = Depends(get_db)):
    j = _job(db, pid, jid)
    trainer.refresh_status(db, j)
    return {"status": j.status, "log": trainer.tail(j, lines)}


@router.get("/{jid}/plot/{name}")
def job_plot(pid: int, jid: int, name: str, db: Session = Depends(get_db)):
    _job(db, pid, jid)   # 404s if this job isn't in this project
    filename = PLOT_FILES.get(name)
    if not filename:
        raise HTTPException(404, "Unknown plot")
    path = RUNS_DIR / f"job_{jid}" / filename
    if not path.exists():
        raise HTTPException(404, "Plot not available yet — it's written once training finishes.")
    return FileResponse(path)


@router.post("/{jid}/stop")
def stop_job(pid: int, jid: int, db: Session = Depends(get_db)):
    j = _job(db, pid, jid)
    trainer.stop(db, j)
    return serialize_job(j)


@router.delete("/{jid}")
def delete_job(pid: int, jid: int, db: Session = Depends(get_db)):
    j = _job(db, pid, jid)
    trainer.delete(db, j)
    return {"ok": True}


def _job(db: Session, pid: int, jid: int) -> TrainJob:
    j = db.get(TrainJob, jid)
    if not j or j.project_id != pid:
        raise HTTPException(404, "Job not found")
    return j
