from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..services import sam, storage
from .images import _img

router = APIRouter(prefix="/api", tags=["sam"])


class SamPrompt(BaseModel):
    type: str = Field(pattern="^(point|box)$")
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] = []


class SamRequest(BaseModel):
    model_name: str = "sam3.pt"
    prompt: SamPrompt
    device: str = "auto"
    simplify: float = 0.01     # polygon simplification (fraction of perimeter)
    max_results: int = 20


@router.get("/sam/models")
def sam_models():
    return sam.list_models()


@router.post("/projects/{pid}/images/{iid}/sam")
def sam_predict(pid: int, iid: int, req: SamRequest, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    path = storage.image_path(pid, im.filename)
    if not path.exists():
        raise HTTPException(404, "Image file missing on disk")
    try:
        results = sam.predict(str(path), req.model_name, req.prompt.model_dump(),
                              req.device, req.simplify)
    except sam.SamError as e:
        raise HTTPException(422, str(e))
    return {"results": results[: req.max_results]}
