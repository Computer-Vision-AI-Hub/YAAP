"""Standalone training worker — launched by the API as a subprocess so the
web server stays responsive and jobs can be stopped by PID.

Usage (internal):
    python -m backend.services.train_worker --job <id>
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import RUNS_DIR                      # noqa: E402
from backend.db import SessionLocal, TrainJob, ModelWeight, init_db  # noqa: E402


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

        model = ModelCls(job.model_arch)
        results = model.train(
            data=params["data"],
            epochs=int(params.get("epochs", 100)),
            imgsz=int(params.get("imgsz", 640)),
            batch=int(params.get("batch", 16)),
            patience=int(params.get("patience", 50)),
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
            # arch is free text read back by is_rtdetr_arch() for future auto-label/train
            # calls on this weight — job.model_arch is a resolved path for continued
            # training and may not contain "rtdetr", so tag it explicitly when it applies.
            arch_label = f"rtdetr:{Path(job.model_arch).name}" if is_rtdetr else job.model_arch
            db.add(ModelWeight(project_id=job.project_id,
                               name=f"{job.model_arch.replace('.pt','')} · job {job.id} (best)",
                               arch=arch_label, task=params.get("task", "detect"),
                               path=str(best), source="trained"))
            db.commit()
        print("YAAP: training finished OK")
    except Exception:
        traceback.print_exc()
        job.status = "failed"
        db.commit()
        sys.exit(1)


if __name__ == "__main__":
    main()
