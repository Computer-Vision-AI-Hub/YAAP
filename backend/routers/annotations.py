import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import Annotation, ImageAsset, LabelClass, get_db
from ..schemas import AnnotationsReplace
from .images import _img

router = APIRouter(prefix="/api/projects/{pid}/images/{iid}/annotations", tags=["annotations"])


def serialize_ann(a: Annotation) -> dict:
    return {"id": a.id, "class_id": a.class_id, "kind": a.kind, "data": a.data_obj,
            "source": a.source, "confidence": a.confidence}


@router.get("")
def get_annotations(pid: int, iid: int, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    return [serialize_ann(a) for a in im.annotations]


@router.put("")
def replace_annotations(pid: int, iid: int, body: AnnotationsReplace,
                        db: Session = Depends(get_db)):
    """The editor autosaves by replacing the full annotation set of an image."""
    im = _img(db, pid, iid)
    valid_classes = {c.id for c in db.query(LabelClass).filter_by(project_id=pid)}
    db.query(Annotation).filter(Annotation.image_id == iid).delete()
    for a in body.annotations:
        if a.class_id not in valid_classes:
            raise HTTPException(400, f"Unknown class id {a.class_id}")
        db.add(Annotation(image_id=iid, class_id=a.class_id, kind=a.kind,
                          data=json.dumps(a.data), source=a.source, confidence=a.confidence))
    im.status = "annotated" if body.annotations else "unannotated"
    db.commit()
    db.refresh(im)
    return {"annotations": [serialize_ann(a) for a in im.annotations], "status": im.status}
