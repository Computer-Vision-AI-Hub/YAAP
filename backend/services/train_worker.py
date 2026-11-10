"""Standalone training worker — launched by the API as a subprocess so the
web server stays responsive and jobs can be stopped by PID.

Usage (internal):
    python -m backend.services.train_worker --job <id>
"""
import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import RUNS_DIR                      # noqa: E402
from backend.db import SessionLocal, TrainJob, ModelWeight, init_db  # noqa: E402


def run_ssh(db, job, params):
    from backend.services import ssh_runner
    try:
        job.status = "running"
        db.commit()
        best, error = ssh_runner.run_job(job, params, db)
        if error:
            raise RuntimeError(error)
        job.weights_path = str(best)
        job.status = "done"
        db.commit()
        arch_label = f"rtdetr:{Path(job.model_arch).name}" if params.get("is_rtdetr") else job.model_arch
        db.add(ModelWeight(project_id=job.project_id,
                           name=f"{job.model_arch.replace('.pt','')} · job {job.id} (best, ssh)",
                           arch=arch_label, task=params.get("task", "detect"),
                           path=str(best), source="trained"))
        db.commit()
        print("YAAP: training finished OK")
    except Exception:
        traceback.print_exc()
        db.refresh(job)
        if job.status != "stopped":
            job.status = "failed"
            db.commit()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, required=True)
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    job = db.get(TrainJob, args.job)
    if not job:
        print(f"job {args.job} not found", file=sys.stderr)
        sys.exit(1)

    params = json.loads(job.params)
    if params.get("mode") == "ssh":
        run_ssh(db, job, params)
        return
    bootstrap_run = params.get("bootstrap_run")     # Bootstrap: label the rest once trained
    weight_id = None
    try:
        import torch
        from ultralytics import RTDETR, YOLO

        is_rtdetr = params.get("is_rtdetr", False)
        ModelCls = RTDETR if is_rtdetr else YOLO

        device = params.get("device", "auto")
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"

        job.status = "running"
        db.commit()

        from backend.services.inference import alias_path, use_models_dir
        use_models_dir()                             # preloaded weights in data/models are found offline
        model = ModelCls(alias_path(job.model_arch))
        if bootstrap_run:
            from backend.services import bootstrap
            bootstrap.track_epochs(model, db, bootstrap_run)
        results = model.train(
            data=params["data"],
            epochs=int(params.get("epochs", 60)),
            imgsz=int(params.get("imgsz", 640)),
            batch=int(params.get("batch", 16)),
            patience=int(params.get("patience", 10)),
            device=device,
            project=str(RUNS_DIR),
            name=f"job_{job.id}",
            exist_ok=True,
            **params.get("extra", {}),
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        job.weights_path = str(best) if best.exists() else ""
        job.status = "done"
        db.commit()

        if best.exists():
            arch_label = f"rtdetr:{Path(job.model_arch).name}" if is_rtdetr else job.model_arch
            w = ModelWeight(project_id=job.project_id,
                            name=params.get("weight_name") or f"{job.model_arch.replace('.pt','')} · job {job.id} (best)",
                            arch=arch_label, task=params.get("task", "detect"),
                            path=str(best), source="trained")
            db.add(w)
            db.commit()
            weight_id = w.id
        print("YAAP: training finished OK")
    except Exception:
        traceback.print_exc()
        job.status = "failed"
        db.commit()
        if bootstrap_run:
            from backend.services import bootstrap
            bootstrap.fail(db, bootstrap_run, "Training failed — see the training log on the Train page.")
        sys.exit(1)

    if bootstrap_run:
        from backend.services import bootstrap
        del model, results                      # free the trainer's GPU memory for inference
        gc.collect()
        torch.cuda.empty_cache()
        try:
            bootstrap.label_phase(db, bootstrap_run, weight_id)
            print("YAAP: bootstrap labeling finished OK")
        except Exception as e:
            traceback.print_exc()
            bootstrap.fail(db, bootstrap_run, f"Labeling failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:                                    # a shutdown deferred by this job can go ahead now
        from backend.services import jobs
        jobs.job_finished()
