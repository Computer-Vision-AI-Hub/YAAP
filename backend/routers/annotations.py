import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import update
from sqlalchemy.orm import Session

from ..db import Annotation, ImageAsset, LabelClass, get_db
from ..schemas import AnnotationsReplace
from .images import _img

router = APIRouter(prefix="/api/projects/{pid}/images/{iid}/annotations", tags=["annotations"])


def serialize_ann(a: Annotation) -> dict:
    return {"id": a.id, "class_id": a.class_id, "kind": a.kind, "data": a.data_obj,
            "source": a.source, "confidence": a.confidence, "origin": a.origin or ""}


@router.get("")
def get_annotations(pid: int, iid: int, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    return [serialize_ann(a) for a in im.annotations]


@router.put("")
def replace_annotations(pid: int, iid: int, body: AnnotationsReplace,
                        db: Session = Depends(get_db)):
    """The editor autosaves by replacing the full annotation set of an image.
    With expect_ids the save is refused (409) if the image no longer holds exactly the annotations the
    client last saw — a background job or another tab changed it; the editor then merges and retries.
    Boxes of a class deleted meanwhile are dropped (their class's other boxes are already gone) instead
    of refusing the whole save; the answer says how many."""
    im = _img(db, pid, iid)
    if body.expect_ids is not None:
        # the no-op UPDATE takes the write lock first, so nothing can change between check and write
        db.execute(update(ImageAsset).where(ImageAsset.id == iid).values(status=ImageAsset.status)
                   .execution_options(synchronize_session=False))
        db.expire(im)
        if sorted(a.id for a in im.annotations) != sorted(body.expect_ids):
            db.rollback()
            raise HTTPException(409, "This image was changed elsewhere (a job or another tab) — merging.")
    task = im.project.task_type
    if task == "classify":
        if any(a.kind != "classification" for a in body.annotations):
            raise HTTPException(422, "A classification project stores one class tag per image — no boxes or outlines.")
        if len(body.annotations) > 1:
            raise HTTPException(422, "A classification image has one class tag, not several.")
    elif any(a.kind == "classification" for a in body.annotations):
        raise HTTPException(422, f"A {task} project stores boxes/outlines, not image class tags.")
    valid_classes = {c.id for c in db.query(LabelClass).filter_by(project_id=pid)}
    keep = [a for a in body.annotations if a.class_id in valid_classes]
    db.query(Annotation).filter(Annotation.image_id == iid).delete()
    for a in keep:
        db.add(Annotation(image_id=iid, class_id=a.class_id, kind=a.kind,
                          data=json.dumps(a.data), source=a.source, confidence=a.confidence,
                          origin=a.origin if a.source == "model" else ""))
    im.status = "annotated" if keep else "unannotated"
    db.commit()
    db.refresh(im)
    # same order as sent (minus dropped ones), so the client can map the new ids onto its shapes
    anns = sorted(im.annotations, key=lambda a: a.id)
    return {"annotations": [serialize_ann(a) for a in anns], "status": im.status,
            "dropped_unknown_class": len(body.annotations) - len(keep)}
