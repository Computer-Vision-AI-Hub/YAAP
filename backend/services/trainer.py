"""Launch / monitor / stop ultralytics training jobs."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from ..config import BASE_DIR, RUNS_DIR
from ..db import DatasetVersion, ModelWeight, Project, TrainJob
from . import exporter


def start(db, project: Project, version: DatasetVersion, req) -> TrainJob:
    from .inference import is_rtdetr_arch, resolve_weights
    is_rtdetr = is_rtdetr_arch(req.model_arch, db)   # check before resolve_weights loses the alias
    arch = resolve_weights(req.model_arch, db)  # 'weight:<id>' → path on disk
    vdir = exporter.version_dir(project.id, version.id)
    if project.task_type == "classify":
        data = str(vdir)                       # classify: dataset root folder
    else:
        data = str(vdir / "data.yaml")
        if not Path(data).exists():
            raise RuntimeError("Version has no data.yaml — regenerate the version.")

    params = {"data": data, "epochs": req.epochs, "imgsz": req.imgsz, "batch": req.batch,
              "patience": req.patience, "device": req.device, "task": project.task_type,
              "is_rtdetr": is_rtdetr, "extra": req.extra}
    job = TrainJob(project_id=project.id, version_id=version.id,
                   model_arch=arch, params=json.dumps(params))
    db.add(job)
    db.commit()

    log_path = RUNS_DIR / f"job_{job.id}.log"
    job.log_path = str(log_path)

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.services.train_worker", "--job", str(job.id)],
        cwd=str(BASE_DIR), stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.pid = proc.pid
    job.status = "running"
    db.commit()
    return job


def tail(job: TrainJob, lines: int = 80) -> str:
    p = Path(job.log_path or "")
    if not p.exists():
        return ""
    content = p.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def stop(db, job: TrainJob):
    if job.pid and job.status == "running":
        try:
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    job.status = "stopped"
    db.commit()


def refresh_status(db, job: TrainJob):
    """If the worker died without updating the DB, mark it failed."""
    if job.status == "running" and job.pid:
        try:
            os.kill(job.pid, 0)
        except ProcessLookupError:
            db.refresh(job)
            if job.status == "running":
                job.status = "failed"
                db.commit()
    return job


def delete(db, job: TrainJob):
    """Remove a job's DB row, its log file, AND its run folder under
    RUNS_DIR (weights, results.png, confusion_matrix.png, etc.)."""
    if job.status == "running":
        stop(db, job)
    if job.log_path:
        Path(job.log_path).unlink(missing_ok=True)
    if job.weights_path:
        db.query(ModelWeight).filter(ModelWeight.path == job.weights_path).delete(synchronize_session=False)
    shutil.rmtree(RUNS_DIR / f"job_{job.id}", ignore_errors=True)
    db.delete(job)
    db.commit()
