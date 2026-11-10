import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..db import Annotation, AutolabelJob, ImageAsset, Proposal, SessionLocal, get_db
from ..services import jobs, sam, storage
from .images import _img
from .models import serialize_autolabel_job
from .projects import project_or_404

router = APIRouter(prefix="/api", tags=["sam"])


class SamPrompt(BaseModel):
    type: str = Field(pattern="^(point|box|exemplar)$")
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] = []


class SamRequest(BaseModel):
    model_name: str = "sam3.pt"
    prompt: SamPrompt
    device: str = "auto"
    simplify: float = 0.01     # polygon simplification (fraction of perimeter)
    max_results: int = 20
    conf: float = 0.25         # exemplar only: minimum match score


@router.get("/sam/models")
def sam_models():
    return sam.list_models()


class WarmupRequest(BaseModel):
    model_name: str = "sam3.pt"
    device: str = "auto"


@router.post("/sam/warmup")
def sam_warmup(req: WarmupRequest):
    """Load SAM 3 ahead of the first prompt so the first click/E isn't slow."""
    try:
        return sam.warmup(req.model_name, req.device)
    except sam.SamError as e:
        raise HTTPException(422, str(e))


@router.post("/projects/{pid}/images/{iid}/sam")
def sam_predict(pid: int, iid: int, req: SamRequest, db: Session = Depends(get_db)):
    im = _img(db, pid, iid)
    path = storage.image_path(pid, im.filename)
    if not path.exists():
        raise HTTPException(404, "Image file missing on disk")
    try:
        if req.prompt.type == "exemplar":
            results = sam.predict_all(str(path), req.model_name, req.prompt.box or None,
                                      conf=req.conf, device=req.device, simplify=req.simplify)
            return {"results": sorted(results, key=lambda r: -r["score"])[:max(req.max_results, 100)]}
        results = sam.predict(str(path), req.model_name, req.prompt.model_dump(),
                              req.device, req.simplify)
    except sam.SamError as e:
        raise HTTPException(422, str(e))
    return {"results": results[: req.max_results]}


class FindTarget(BaseModel):
    class_id: int
    text: str = Field(min_length=1, max_length=80)


class FindAllRequest(BaseModel):
    """Dataset-wide concept search: SAM 3 finds every instance of text phrases in
    many images — one class per phrase, several classes in one run. (A box exemplar
    only makes sense inside its own image.)"""
    model_name: str = "sam3.pt"
    mode: str = Field("classes", pattern="^(classes|generic)$")
    # generic: no classes — SAM outlines every object, saved as class-less proposals the
    # editor then walks through for labeling. `phrases` are generic nouns, not class names.
    phrases: list[str] = []
    targets: list[FindTarget] = []     # classes mode: [{class_id, text}, ...]
    class_id: Optional[int] = None     # single-target shorthand, same as targets=[{class_id, text}]
    text: str = ""
    image_ids: list[int] = []          # empty = all images
    only_unannotated: bool = True
    conf: float = 0.4
    max_overlap: float = Field(0.7, gt=0, le=1)   # one box per object, see jobs.drop_duplicates
    device: str = "auto"
    simplify: float = 0.008

    def all_targets(self) -> list[FindTarget]:
        if self.targets:
            return self.targets
        if self.class_id is not None and self.text.strip():
            return [FindTarget(class_id=self.class_id, text=self.text.strip())]
        return []


@router.post("/projects/{pid}/sam/find-all")
def sam_find_all(pid: int, req: FindAllRequest, background: BackgroundTasks,
                 db: Session = Depends(get_db)):
    p = project_or_404(db, pid)
    if p.task_type == "classify":
        raise HTTPException(400, "Find-all produces boxes/polygons — not available for classification projects.")
    if req.model_name not in sam.SAM_MODELS:
        raise HTTPException(400, f"Unknown SAM model '{req.model_name}'.")
    targets = req.all_targets()
    if req.mode == "generic":
        if not [x for x in req.phrases if x.strip()] and req.phrases:
            raise HTTPException(400, "Give at least one generic phrase.")
    else:
        if not targets:
            raise HTTPException(400, "Pick at least one class and phrase.")
        class_ids = {c.id for c in p.classes}
        if any(tg.class_id not in class_ids for tg in targets):
            raise HTTPException(400, "Unknown class")
    images = [im for im in p.images
              if (not req.image_ids or im.id in req.image_ids)
              and (not req.only_unannotated or not im.annotations)]
    if not images:
        raise HTTPException(400, "No images match — nothing to search.")
    label = ("generic objects" if req.mode == "generic" else
             f"“{targets[0].text}”" if len(targets) == 1 else f"{len(targets)} classes")
    job = AutolabelJob(project_id=pid, model_name=f"{req.model_name} · {label}",
                       total=len(images), status="running")
    db.add(job); db.commit()
    background.add_task(_run_find_all, pid, job.id, [im.id for im in images], req)
    return serialize_autolabel_job(job)


GENERIC_PHRASES = ["component", "part", "object"]   # SAM 3 answers noun phrases; bare "object" alone often finds little


def _run_generic(pid: int, jid: int, image_ids: list[int], req: FindAllRequest):
    """Outline everything, assign no class: results are Proposals, not Annotations."""
    db = SessionLocal()
    try:
        job = db.get(AutolabelJob, jid)
        # no nouns given: the project's class names say best what the images show ("car", "bus" find
        # vehicles, "object" doesn't); the generic nouns only for a project without classes
        phrases = ([x.strip() for x in req.phrases if x.strip()]
                   or [c.name for c in project_or_404(db, pid).classes] or GENERIC_PHRASES)
        skipped, labeled, log = [], [], []
        for n, iid in enumerate(image_ids, 1):
            im = db.get(ImageAsset, iid)
            if im is None:
                job.processed += 1
                continue
            path = storage.image_path(pid, im.filename)
            try:
                if not path.exists():
                    raise sam.SamError("file missing")
                by_phrase = sam.predict_phrases(str(path), req.model_name, phrases,
                                                conf=req.conf, device=req.device, simplify=req.simplify)
            except sam.SamError as e:
                skipped.append(im.filename)
                log.append(f"[{n}/{job.total}] {im.filename} — failed: {e}")
                by_phrase = None
            if by_phrase is not None:
                try:
                    # union of all phrases, one outline per object; objects already annotated are skipped
                    existing = [jobs.shape(a.class_id, a.kind, a.data_obj) for a in im.annotations]
                    kept, _ = jobs.drop_duplicates(
                        [{"cls": None, "box": r["bbox"], "poly": r["polygon"], "conf": r["score"], "r": r}
                         for rs in by_phrase.values() for r in rs], existing, req.max_overlap)
                    kept = [k["r"] for k in kept]
                    db.query(Proposal).filter(Proposal.image_id == im.id).delete()
                    for r in kept:
                        db.add(Proposal(image_id=im.id, score=r["score"],
                                        data=json.dumps({"polygon": r["polygon"], "bbox": r["bbox"]})))
                    db.commit()
                    if kept:
                        labeled.append(im.id)
                        job.images_labeled += 1
                        job.annotations_added += len(kept)
                    log.append(f"[{n}/{job.total}] {im.filename} — {len(kept)} object(s) outlined")
                except OperationalError as e:          # e.g. "database is locked": skip this image only
                    db.rollback()
                    job = db.get(AutolabelJob, jid)
                    skipped.append(im.filename)
                    log.append(f"[{n}/{job.total}] {im.filename} — not saved, database busy ({e.orig}); "
                               "run again to retry")
            job.processed += 1
            job.skipped = json.dumps(skipped)
            job.labeled_image_ids = json.dumps(labeled)
            job.log = "\n".join(log)
            db.commit()
        job.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        job = db.get(AutolabelJob, jid)
        if job:
            job.status, job.error = "failed", str(e)
            db.commit()
    finally:
        db.close()
        jobs.job_finished()


def _run_find_all(pid: int, jid: int, image_ids: list[int], req: FindAllRequest):
    if req.mode == "generic":
        return _run_generic(pid, jid, image_ids, req)
    db = SessionLocal()
    try:
        job = db.get(AutolabelJob, jid)
        task = project_or_404(db, pid).task_type
        targets = req.all_targets()
        skipped, labeled, log = [], [], []
        # what the job saw; a change means a human edited the image meanwhile
        seen = {im.id: im.status for im in db.query(ImageAsset).filter(ImageAsset.id.in_(image_ids))}
        for n, iid in enumerate(image_ids, 1):
            im = db.get(ImageAsset, iid)
            if im is None:
                job.processed += 1
                continue
            path = storage.image_path(pid, im.filename)
            try:
                if not path.exists():
                    raise sam.SamError("file missing")
                by_phrase = sam.predict_phrases(str(path), req.model_name, [tg.text for tg in targets],
                                                conf=req.conf, device=req.device, simplify=req.simplify)
            except sam.SamError as e:
                skipped.append(im.filename)
                log.append(f"[{n}/{job.total}] {im.filename} — failed: {e}")
                by_phrase = None
            try:
                _save_matches(db, job, im, n, by_phrase, seen[iid], targets, task, req, labeled, log)
                db.commit()
            except OperationalError as e:               # e.g. "database is locked": skip this image only
                db.rollback()
                if labeled and labeled[-1] == iid:
                    labeled.pop()
                job = db.get(AutolabelJob, jid)
                skipped.append(im.filename)
                log.append(f"[{n}/{job.total}] {im.filename} — not saved, database busy ({e.orig}); "
                           "run again to retry")
            job.processed += 1
            job.skipped = json.dumps(skipped)
            job.labeled_image_ids = json.dumps(labeled)
            job.log = "\n".join(log)
            db.commit()
        job.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        job = db.get(AutolabelJob, jid)
        if job:
            job.status, job.error = "failed", str(e)
            db.commit()
    finally:
        db.close()
        jobs.job_finished()


def _save_matches(db, job, im, n, by_phrase, seen_status, targets, task, req, labeled, log):
    """Write one image's find-all matches (under the write lock taken by claim_image)."""
    iid = im.id
    if by_phrase is not None and any(by_phrase.values()) and not jobs.claim_image(db, iid, seen_status):
        db.rollback()
        log.append(f"[{n}/{job.total}] {im.filename} — changed by hand meanwhile, left alone")
        by_phrase = None
    if by_phrase is not None:
        db.expire(im, ["annotations"])          # re-read under the write lock
        existing = [jobs.shape(a.class_id, a.kind, a.data_obj) for a in im.annotations]
        # one box per object: two phrases hitting the same object, or an object already labeled
        # (segment projects store the outline, so they compare outlines)
        kept, dropped = jobs.drop_duplicates(
            [{"cls": tg.class_id, "box": r["bbox"], "poly": r["polygon"] if task == "segment" else None,
              "conf": r["score"], "r": r}
             for tg in targets for r in by_phrase.get(tg.text.strip(), [])], existing, req.max_overlap)
        added = 0
        for k in kept:
            r = k["r"]
            kind, data = ("bbox", r["bbox"]) if task == "detect" else ("polygon", {"points": r["polygon"]})
            db.add(Annotation(image_id=im.id, class_id=k["cls"], kind=kind,
                              data=json.dumps(data), source="model", confidence=r["score"], origin="sam"))
            added += 1
        if added:
            im.status = "auto-annotated"
            labeled.append(im.id)
            job.images_labeled += 1
            job.annotations_added += added
        log.append(f"[{n}/{job.total}] {im.filename} — {added} match(es)"
                   + (f", {dropped} duplicate(s) skipped" if dropped else ""))
