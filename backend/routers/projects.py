import shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import LabelClass, Project, get_db
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


def serialize_project(p: Project) -> dict:
    n_ann = sum(1 for im in p.images if im.status in ("annotated", "empty") or im.annotations)
    return {"id": p.id, "name": p.name, "description": p.description,
            "task_type": p.task_type, "created_at": p.created_at.isoformat(),
            "image_count": len(p.images), "annotated_count": n_ann,
            "class_count": len(p.classes), "version_count": len(p.versions),
            "classes": [serialize_class(c) for c in p.classes]}


def serialize_class(c: LabelClass) -> dict:
    return {"id": c.id, "name": c.name, "color": c.color, "order_idx": c.order_idx}


@router.get("")
def list_projects(db: Session = Depends(get_db)):
    return [serialize_project(p) for p in db.query(Project).order_by(Project.created_at.desc())]


@router.post("")
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(name=body.name.strip() or "Untitled", description=body.description,
                task_type=body.task_type)
    db.add(p); db.commit()
    storage.project_dir(p.id)
    return serialize_project(p)


@router.get("/{pid}")
def get_project(pid: int, db: Session = Depends(get_db)):
    return serialize_project(project_or_404(db, pid))


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
