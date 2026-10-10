import shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, exists, func, or_
from sqlalchemy.orm import Session, selectinload

from ..db import Annotation, DatasetVersion, ImageAsset, LabelClass, Project, get_db
from ..schemas import ClassCreate, ClassUpdate, ProjectCreate
from ..services import storage

router = APIRouter(prefix="/api/projects", tags=["projects"])

PALETTE = ["#FFC53D", "#4CC38A", "#52A9FF", "#E5484D", "#BF7AF0",
           "#F76808", "#1FD8A4", "#FF8DCC", "#96C7F2", "#F0C000"]


def project_or_404(db: Session, pid: int) -> Project:
    p = db.get(Project, pid)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


def _project_counts(db: Session, pids: list[int]) -> dict[int, dict]:
    """Per-project image/labeled/version counts, aggregated in SQL — a fixed
    number of queries no matter how many projects or images there are."""
    out = {pid: {"images": 0, "annotated": 0, "versions": 0} for pid in pids}
    labeled = or_(ImageAsset.status.in_(("annotated", "empty")),
                  exists().where(Annotation.image_id == ImageAsset.id))
    for pid, n, n_ann in (db.query(ImageAsset.project_id, func.count(ImageAsset.id),
                                   func.sum(case((labeled, 1), else_=0)))
                          .filter(ImageAsset.project_id.in_(pids))
                          .group_by(ImageAsset.project_id)):
        out[pid].update(images=n, annotated=int(n_ann or 0))
    for pid, n in (db.query(DatasetVersion.project_id, func.count(DatasetVersion.id))
                   .filter(DatasetVersion.project_id.in_(pids))
                   .group_by(DatasetVersion.project_id)):
        out[pid]["versions"] = n
    return out


def serialize_projects(db: Session, projects: list[Project]) -> list[dict]:
    counts = _project_counts(db, [p.id for p in projects])
    return [{"id": p.id, "name": p.name, "description": p.description,
             "task_type": p.task_type, "created_at": p.created_at.isoformat(),
             "image_count": counts[p.id]["images"], "annotated_count": counts[p.id]["annotated"],
             "class_count": len(p.classes), "version_count": counts[p.id]["versions"],
             "classes": [serialize_class(c) for c in p.classes]} for p in projects]


def serialize_class(c: LabelClass) -> dict:
    return {"id": c.id, "name": c.name, "color": c.color, "order_idx": c.order_idx}


@router.get("")
def list_projects(db: Session = Depends(get_db)):
    projects = (db.query(Project).options(selectinload(Project.classes))
                .order_by(Project.created_at.desc()).all())
    return serialize_projects(db, projects)


@router.post("")
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(name=body.name.strip() or "Untitled", description=body.description,
                task_type=body.task_type)
    db.add(p); db.commit()
    storage.project_dir(p.id)
    return serialize_projects(db, [p])[0]


@router.get("/{pid}")
def get_project(pid: int, db: Session = Depends(get_db)):
    return serialize_projects(db, [project_or_404(db, pid)])[0]


@router.delete("/{pid}")
def delete_project(pid: int, db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    shutil.rmtree(storage.project_dir(pid), ignore_errors=True)
    db.delete(p); db.commit()
    return {"ok": True}


# ── classes ──────────────────────────────────────────────────────────
@router.post("/{pid}/classes")
def add_class(pid: int, body: ClassCreate, db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Class name is empty")
    if any(c.name == name for c in p.classes):
        raise HTTPException(409, f"Class '{name}' already exists")
    color = body.color or PALETTE[len(p.classes) % len(PALETTE)]
    c = LabelClass(project_id=pid, name=name, color=color, order_idx=len(p.classes))
    db.add(c); db.commit()
    return serialize_class(c)


@router.patch("/{pid}/classes/{cid}")
def update_class(pid: int, cid: int, body: ClassUpdate, db: Session = Depends(get_db)):
    c = db.get(LabelClass, cid)
    if not c or c.project_id != pid:
        raise HTTPException(404, "Class not found")
    if body.name is not None:
        c.name = body.name.strip() or c.name
    if body.color is not None:
        c.color = body.color
    db.commit()
    return serialize_class(c)


@router.delete("/{pid}/classes/{cid}")
def delete_class(pid: int, cid: int, db: Session = Depends(get_db)):
    from ..db import Annotation
    c = db.get(LabelClass, cid)
    if not c or c.project_id != pid:
        raise HTTPException(404, "Class not found")
    db.query(Annotation).filter(Annotation.class_id == cid).delete()
    db.delete(c); db.commit()
    return {"ok": True}
