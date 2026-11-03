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
from . import exporter, ssh_runner


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
    if req.mode == "ssh":
        settings = ssh_runner.load_settings()
        try:
            ssh_runner.validate(settings)
        except ssh_runner.SshError as e:
            raise RuntimeError(str(e))
        params.update(mode="ssh", ssh=settings, local_data_dir=str(vdir))
    job = TrainJob(project_id=project.id, version_id=version.id,
                   model_arch=arch, params=json.dumps(params))
    db.add(job)
    db.commit()

    if req.mode == "ssh":      # job id is only known after the insert
        params["local_run_dir"] = str(RUNS_DIR / f"job_{job.id}")
        job.params = json.dumps(params)
    return launch(db, job)


def launch(db, job: TrainJob) -> TrainJob:
    """Start train_worker for an already inserted job; output goes to RUNS_DIR/job_<id>.log."""
    log_path = RUNS_DIR / f"job_{job.id}.log"
    job.log_path = str(log_path)

    # the child inherits its own copy of the descriptor, so ours can close now
    with open(log_path, "w") as log_file:
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


_HAS_PROC = Path("/proc/self/cmdline").exists()


def is_worker(pid: int, job_id: int) -> bool | None:
    """Is `pid` this job's train_worker? After a container restart pids start over, so
    a stored pid can belong to anything — even the server. None where /proc can't tell."""
    if not _HAS_PROC:
        return None
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return (b"backend.services.train_worker" in args and b"--job" in args
            and args[args.index(b"--job") + 1:][:1] == [str(job_id).encode()])


def worker_alive(job: TrainJob) -> bool:
    pid = job.pid
    if not pid or pid <= 0:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0]:     # our child, finished: reap it
            return False
    except ChildProcessError:
        pass                                   # not our child (server restarted)
    known = is_worker(pid, job.id)
    if known is not None:
        return known
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def kill_worker(job: TrainJob):
    """SIGTERM the worker's process group (dataloader workers included) — only if it
    still is that worker."""
    if job.pid and job.pid > 0 and is_worker(job.pid, job.id) is not False:
        try:
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def _active_bootstrap(db, job: TrainJob):
    rid = job_params(job).get("bootstrap_run")
    if not rid:
        return None
    from . import bootstrap                    # imports this module
    from ..db import BootstrapRun
    run = db.get(BootstrapRun, rid)
    return run if run and run.status in bootstrap.ACTIVE else None


def stop(db, job: TrainJob):
    run = _active_bootstrap(db, job)
    if run:                                    # the same worker also labels afterwards
        from . import bootstrap
        bootstrap.stop(db, run)
        return
    if job.status == "running":
        kill_worker(job)
    _ssh_call(job, "kill")
    job.status = "stopped"
    db.commit()


def job_params(job: TrainJob) -> dict:
    try:
        return json.loads(job.params or "{}")
    except ValueError:
        return {}


def ssh_info(job: TrainJob) -> dict | None:
    p = job_params(job)
    return p if p.get("mode") == "ssh" else None


def _ssh_call(job: TrainJob, what: str):
    """Best-effort remote kill / folder removal; never blocks stop/delete on ssh errors."""
    p = ssh_info(job)
    if not p:
        return
    try:
        if what == "kill" and p.get("remote_pid"):
            ssh_runner.kill_remote(p["ssh"], p["remote_pid"])
        elif what == "remove" and p.get("remote_dir"):
            ssh_runner.remove_remote(p["ssh"], p["remote_dir"])
    except ssh_runner.SshError:
        pass


def refresh_status(db, job: TrainJob):
    """If the worker died without updating the DB, mark it failed."""
    if job.status == "running" and job.pid and not worker_alive(job):
        db.refresh(job)
        if job.status == "running":
            job.status = "failed"
            db.commit()
    return job


def delete(db, job: TrainJob, remote: bool = False):
    """Remove a job's DB row, its log file, AND its run folder under
    RUNS_DIR (weights, results.png, confusion_matrix.png, etc.)."""
    if job.status == "running" or _active_bootstrap(db, job):
        stop(db, job)
    if remote:
        _ssh_call(job, "remove")
    if job.log_path:
        Path(job.log_path).unlink(missing_ok=True)
    if job.weights_path:
        db.query(ModelWeight).filter(ModelWeight.path == job.weights_path).delete(synchronize_session=False)
    shutil.rmtree(RUNS_DIR / f"job_{job.id}", ignore_errors=True)
    db.delete(job)
    db.commit()
