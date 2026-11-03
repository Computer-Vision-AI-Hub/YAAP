"""Standalone training script for YAAP's SSH mode.

Copied to the remote machine, which only needs torch + ultralytics — so this
file must NOT import anything from `backend`. Mirrors train_worker.py's
training call (YOLO / RT-DETR). TODO: RF-DETR — local training doesn't
support it either.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def localize_data_yaml(data: str) -> str:
    """The exported data.yaml's `path:` points at the YAAP machine — repoint it
    at the yaml's own directory. Classify datasets are plain folders (no yaml)."""
    p = Path(data)
    if p.suffix.lower() not in (".yaml", ".yml"):
        return str(p.resolve())
    import yaml
    cfg = yaml.safe_load(p.read_text())
    cfg["path"] = str(p.resolve().parent)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(p.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="detect")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="run")
    ap.add_argument("--rtdetr", action="store_true")
    ap.add_argument("--extra", default="{}")
    a = ap.parse_args()

    import torch
    from ultralytics import RTDETR, YOLO

    device = a.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    project = Path(a.project).resolve()
    model = (RTDETR if a.rtdetr else YOLO)(a.model)
    results = model.train(
        data=localize_data_yaml(a.data),
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
        device=device, project=str(project), name=a.name, exist_ok=True,
        **json.loads(a.extra or "{}"),
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print("YAAP: training finished but best.pt was not found", file=sys.stderr)
        sys.exit(2)
    print("YAAP_RESULT " + json.dumps({"best": str(best.resolve().relative_to(project))}), flush=True)


if __name__ == "__main__":
    main()
