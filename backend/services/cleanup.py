"""Deleting a project, completely: the ORM cascade only covers images, classes and
versions — training jobs, weights, job logs and Bootstrap rounds point at the project
by project_id alone, and their files live outside the project folder."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import MODELS_DIR, RUNS_DIR
from ..db import AutolabelJob, BootstrapRun, ModelWeight, Project, TrainJob
from . import storage, trainer


def delete_project(db, project: Project):
    pid = project.id
    # stops a running one first (a Bootstrap round too), then removes runs/job_<id>/,
    # its log and the weight it registered
    for job in db.query(TrainJob).filter_by(project_id=pid).all():
        trainer.delete(db, job)
    for w in db.query(ModelWeight).filter_by(project_id=pid).all():   # uploaded for this project
        _remove_weight_file(db, w)
        db.delete(w)
    db.query(AutolabelJob).filter_by(project_id=pid).delete(synchronize_session=False)
    db.query(BootstrapRun).filter_by(project_id=pid).delete(synchronize_session=False)
    shutil.rmtree(storage.project_dir(pid), ignore_errors=True)
    db.delete(project)
    db.commit()


def _remove_weight_file(db, w: ModelWeight):
    """Only files YAAP stored itself, and only when no other weight row still uses them."""
    path = Path(w.path)
    try:
        resolved = path.resolve()
    except OSError:
        return
    ours = any(resolved.is_relative_to(d.resolve()) for d in (MODELS_DIR, RUNS_DIR))
    if ours and not db.query(ModelWeight).filter(ModelWeight.path == w.path, ModelWeight.id != w.id).first():
        path.unlink(missing_ok=True)
