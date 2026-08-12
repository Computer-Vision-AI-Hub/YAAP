import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import MODELS_DIR, PRETRAINED_MODELS, RFDETR_VARIANTS, RTDETR_VARIANTS
from ..db import Annotation, ImageAsset, LabelClass, ModelWeight, get_db
from ..schemas import InferenceRequest
from ..services import inference, storage
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


@router.post("/projects/{pid}/autolabel")
def autolabel(pid: int, req: InferenceRequest, db: Session = Depends(get_db)):
    """Run a model over images and store predictions as annotations (source=model)."""
    project = project_or_404(db, pid)
    images = (db.query(ImageAsset).filter(ImageAsset.id.in_(req.image_ids)).all()
              if req.image_ids else list(project.images))
    if not images:
        raise HTTPException(400, "No images to label")

    by_name = {c.name: c for c in project.classes}
    labeled, total_anns, skipped = 0, 0, []

    for im in images:
        path = storage.image_path(pid, im.filename)
        if not path.exists():
            continue
        try:
            preds = inference.predict(req.model_name, str(path), project.task_type,
                                      req.conf, req.iou, req.device, db)
        except inference.InferenceError as e:
            raise HTTPException(422, str(e))
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
            im.status = "review"          # model output waits for a human pass
            labeled += 1
            total_anns += added
        db.commit()

    return {"images_labeled": labeled, "annotations_added": total_anns,
            "images_total": len(images), "skipped": skipped}


@router.get("/system/device")
def device_info():
    return inference.torch_info()
