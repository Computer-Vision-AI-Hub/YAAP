"""Bootstrap — label a handful of images, a quickly trained model labels the rest.

One round: train a small YOLO model on every human-checked image (annotated /
approved, plus images marked empty as background), then run it over the images
nobody has labeled and, optionally, over earlier predictions nobody has reviewed
yet and over rejected ones. Its output lands as model predictions (status
auto-annotated, i.e. in Review); accepting them there makes them training data for
the next round, rejecting them hands them to the round after that.

Both phases run in the train_worker subprocess: training is an ordinary TrainJob
(log, Stop and the registered weights show on the Train page too), and the
labeling pass writes an AutolabelJob so the Auto-label page can show its log.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import yaml
from sqlalchemy import update

from ..config import PRETRAINED_MODELS, RUNS_DIR
from ..db import Annotation, AutolabelJob, BootstrapRun, ImageAsset, Project, TrainJob
from . import exporter, inference, jobs, storage, trainer

ACTIVE = ("training", "labeling")
TRAIN_STATUSES = ("annotated", "approved", "empty")    # a human has checked these
MIN_LABELED = 5
RECOMMENDED = 10        # ~0.74 box F1 on a 5-class test set (spike, 2026-10)
FEW_EXAMPLES = 5        # a class with fewer labeled objects than this gets mixed up with others
# classification needs far more: on VisDrone vehicle crops 8 per class gave ~35 % accuracy (4 classes),
# 24 per class ~55 %, 40 per class ~65 %
FEW_EXAMPLES_CLS = 20
DEFAULT_IMGSZ = {"detect": 640, "segment": 640, "classify": 224}   # 224 = ultralytics' classification size
DEFAULT_MODEL = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt", "classify": "yolo11n-cls.pt"}
LAUNCH_GRACE = 600      # s a round may sit between "created" and "worker started"


class BootstrapError(RuntimeError):
    pass


class BootstrapBusy(BootstrapError):
    pass


def _now():
    return dt.datetime.utcnow()


def auto_epochs(n_images: int) -> int:
    """Few images need more passes: up to 15 images → 300 epochs, 30 → 150, 90 or more → 50.
    (On VisDrone, 15 images at imgsz 1280: 100 epochs gave box F1 .38, 300 gave .51 — the old
    150-epoch cap left small-object models badly under-trained.)"""
    return max(50, min(300, 4500 // max(n_images, 1)))


def auto_batch(n_images: int, imgsz: int) -> int:
    """8 (16 from 64 images on) at 640 px, halved for every doubling of imgsz: GPU memory grows with
    the pixel count, and batch 16 at 1280 ran out of memory on a 6 GB card."""
    b = 8 if n_images < 64 else 16
    while imgsz > 640 * 1.25 and b > 2:
        imgsz /= 2
        b //= 2
    return b


def class_folder(name: str) -> str:
    """Classification trains from one folder per class — the name must be a path segment."""
    return name.replace("/", "_").replace("\\", "_")


def training_images(project: Project) -> list[ImageAsset]:
    """Human-checked images. The ones without annotations train as background
    (detect/segment only — classification needs a class on every image)."""
    out = [im for im in project.images if im.status in TRAIN_STATUSES]
    if project.task_type == "classify":
        out = [im for im in out if any(a.kind == "classification" for a in im.annotations)]
    return out


OWN_ORIGINS = ("bootstrap", "")      # "" = made before origins were recorded: replaced, as it always was


def is_target(im: ImageAsset, refresh_unreviewed: bool, relabel_rejected: bool = False,
              other_tools: bool = False) -> bool:
    """May a round (re)label this image? Yes if nobody labeled it; with refresh_unreviewed
    if all it holds is an unreviewed model guess; with relabel_rejected if it is a model
    guess someone rejected. Boxes a person drew are never replaced, and guesses made by another
    tool (SAM find-all, Run a model) only with other_tools — an early Bootstrap model is often
    worse than SAM, and replacing SAM's boxes threw good work away."""
    if im.status == "unannotated":
        return not im.annotations
    if (im.status == "auto-annotated" and refresh_unreviewed) or \
            (im.status == "needs_reannotation" and relabel_rejected):
        return all(a.source == "model" and (other_tools or (a.origin or "") in OWN_ORIGINS)
                   for a in im.annotations)
    return False


def _other_tools_only(im: ImageAsset) -> bool:
    """Unreviewed/rejected predictions that only relabel_other_tools would let a round replace."""
    return (im.status in ("auto-annotated", "needs_reannotation")
            and is_target(im, True, True, True) and not is_target(im, True, True, False))


def summary(db, project: Project) -> dict:
    """What the Auto-label page needs before (and between) rounds. Counted with grouped SQL queries — going
    through every image's annotations in Python took 4–8 s on a 5 000-image project."""
    from sqlalchemy import case, func
    own = list(OWN_ORIGINS)
    counts = {iid: (n, manual, other, cls) for iid, n, manual, other, cls in db.query(
        Annotation.image_id, func.count(Annotation.id),
        func.sum(case((Annotation.source != "model", 1), else_=0)),
        func.sum(case((Annotation.source == "model", case((func.coalesce(Annotation.origin, "").in_(own), 0), else_=1)),
                      else_=0)),
        func.sum(case((Annotation.kind == "classification", 1), else_=0)))
        .join(ImageAsset, ImageAsset.id == Annotation.image_id).filter(ImageAsset.project_id == project.id)
        .group_by(Annotation.image_id)}
    status = dict(db.query(ImageAsset.id, ImageAsset.status).filter(ImageAsset.project_id == project.id))
    c = lambda iid: counts.get(iid, (0, 0, 0, 0))  # noqa: E731
    train_ids = [i for i, st in status.items() if st in TRAIN_STATUSES
                 and (project.task_type != "classify" or c(i)[3] > 0)]
    labeled = sum(1 for i in train_ids if c(i)[0] > 0)
    per_class = {cl.id: 0 for cl in project.classes}
    for cid, n in (db.query(Annotation.class_id, func.count(Annotation.id))
                   .join(ImageAsset, ImageAsset.id == Annotation.image_id)
                   .filter(ImageAsset.project_id == project.id, ImageAsset.status.in_(list(TRAIN_STATUSES)))
                   .group_by(Annotation.class_id)):
        if cid in per_class:
            per_class[cid] = n
    model_only = lambda i, other_ok: c(i)[1] == 0 and (other_ok or c(i)[2] == 0)  # noqa: E731  (= is_target's rule)
    by_status = lambda st, cond: sum(1 for i, s_ in status.items() if s_ == st and cond(i))  # noqa: E731
    train = train_ids
    names = {c.id: c.name for c in project.classes}
    runs = (db.query(BootstrapRun).filter_by(project_id=project.id)
            .order_by(BootstrapRun.id.desc()).all())
    if runs:
        refresh(db, runs[0])
    last_done = next((r for r in runs if r.status == "done"), None)
    return {
        "task": project.task_type,
        "models": PRETRAINED_MODELS.get(project.task_type, []),
        "default_model": DEFAULT_MODEL.get(project.task_type, ""),
        "labeled": labeled, "background": len(train) - labeled,
        # e.g. 10 Shapes images with only 4 circles: the model finds every shape, but calls many circles hexagons
        "few_examples": [{"name": names[cid], "count": n} for cid, n in per_class.items()
                         if n < (FEW_EXAMPLES_CLS if project.task_type == "classify" else FEW_EXAMPLES)],
        "default_imgsz": DEFAULT_IMGSZ.get(project.task_type, 640),
        "unannotated": by_status("unannotated", lambda i: c(i)[0] == 0),
        "unreviewed": by_status("auto-annotated", lambda i: model_only(i, False)),
        "rejected": by_status("needs_reannotation", lambda i: model_only(i, False)),
        # predictions from SAM / Run a model: kept unless "also replace other tools' predictions" is ticked
        "other_unreviewed": by_status("auto-annotated", lambda i: model_only(i, True) and c(i)[2] > 0),
        "other_rejected": by_status("needs_reannotation", lambda i: model_only(i, True) and c(i)[2] > 0),
        "min_labeled": MIN_LABELED, "recommended": RECOMMENDED,
        "auto_epochs": auto_epochs(len(train)),
        "next_round": _next_round(db, project.id),
        "new_since_last": labeled - last_done.train_images if last_done else None,
        "last_run": runs[0] if runs else None,
    }


def _next_round(db, pid: int) -> int:
    return db.query(BootstrapRun).filter_by(project_id=pid, status="done").count() + 1


def to_review(db, run: BootstrapRun) -> int:
    """How many of this round's images still wait in Review."""
    aj = db.get(AutolabelJob, run.autolabel_job_id) if run.autolabel_job_id else None
    ids = json.loads(aj.labeled_image_ids or "[]") if aj else []
    if not ids:
        return 0
    return (db.query(ImageAsset).filter(ImageAsset.id.in_(ids), ImageAsset.status == "auto-annotated")
            .count())


def start(db, project: Project, req) -> BootstrapRun:
    """Validate, build the training set and launch the worker. Returns the new run."""
    task = project.task_type
    classes = list(project.classes)
    if not classes:
        raise BootstrapError("Add your classes and label a few images first.")
    for r in db.query(BootstrapRun).filter(BootstrapRun.project_id == project.id,
                                           BootstrapRun.status.in_(ACTIVE)).all():
        if refresh(db, r).status in ACTIVE:
            raise BootstrapBusy(f"Bootstrap round {r.round} is still running.")
    arch = req.model_arch or DEFAULT_MODEL.get(task, "")
    if arch not in PRETRAINED_MODELS.get(task, []):
        raise BootstrapError(f"'{arch}' is not a {task} model — pick one from the list.")
    inference.resolve_weights(arch)     # also points ultralytics' downloads at MODELS_DIR, like Train does

    train = training_images(project)
    labeled = [im for im in train if im.annotations]
    if len(labeled) < MIN_LABELED:
        raise BootstrapError(f"Label at least {MIN_LABELED} images first ({len(labeled)} so far) — "
                             f"about {RECOMMENDED} already gives a useful model.")
    if task == "classify" and len({a.class_id for im in labeled for a in im.annotations
                                   if a.kind == "classification"}) < 2:
        raise BootstrapError("Classification needs labeled images of at least two classes.")
    targets = [im for im in project.images
               if is_target(im, req.refresh_unreviewed, req.relabel_rejected, req.relabel_other_tools)]
    if not targets:
        raise BootstrapError("Every image is already labeled or reviewed — nothing left to label.")

    epochs = req.epochs or auto_epochs(len(train))
    imgsz = req.imgsz or DEFAULT_IMGSZ.get(task, 640)
    rnd = _next_round(db, project.id)
    cfg = req.model_dump()
    # the model's class names → class ids, as of now: a class renamed mid-round still maps
    cfg["class_map"] = {**{class_folder(c.name): c.id for c in classes}, **{c.name: c.id for c in classes}}
    run = BootstrapRun(project_id=project.id, round=rnd, status="training", config=json.dumps(cfg),
                       model_arch=arch, epochs=epochs, train_images=len(labeled),
                       target_images=len(targets), message="Preparing the training set…")
    job = TrainJob(project_id=project.id, version_id=0, model_arch=arch)   # no version: own set, below
    db.add_all([run, job])
    db.flush()
    run.train_job_id = job.id
    job.params = json.dumps({"bootstrap_run": run.id, "bootstrap_round": rnd})
    db.commit()                          # job stays "queued" with no pid until launched — see refresh()
    try:
        data = build_dataset(project, train, _dataset_dir(job.id))
    except Exception as e:
        run.status, run.error, run.finished_at = "failed", f"Could not build the training set: {e}", _now()
        db.delete(job)
        db.commit()
        raise BootstrapError(run.error) from e
    job.params = json.dumps({
        "data": str(data), "epochs": epochs, "imgsz": imgsz,
        "batch": auto_batch(len(train), imgsz), "patience": 0, "device": req.device,
        "task": task, "is_rtdetr": False,
        "extra": {"val": False},    # tiny sets: no held-out split, so per-epoch validation is wasted time
        "bootstrap_run": run.id, "bootstrap_round": rnd,
        "weight_name": f"Bootstrap r{rnd} · {Path(arch).stem} · {len(labeled)} imgs",
    })
    run.message = f"Starting training on {len(labeled)} labeled image(s)…"
    db.commit()
    _release_gpu()
    try:
        trainer.launch(db, job)
    except Exception as e:
        run.status, run.error, run.finished_at = "failed", f"Could not start the worker: {e}", _now()
        job.status = "failed"
        db.commit()
        raise BootstrapError(run.error) from e
    return run


def _release_gpu():
    try:
        from . import sam
        sam.release_gpu()
    except Exception:
        pass        # SAM never loaded / not installed — nothing to free


def _dataset_dir(job_id: int) -> Path:
    return RUNS_DIR / f"job_{job_id}" / "dataset"


def build_dataset(project: Project, images: list[ImageAsset], out: Path) -> Path:
    """Write `images` in ultralytics layout under `out`; returns what `data=` should get.
    Originals are COPIED, never linked: ultralytics re-saves JPEGs it thinks are corrupt
    (bytes after the end marker, common on phone photos) in place. Annotations are in the
    image's EXIF-oriented pixel frame, which is also how ultralytics reads it. Validation
    reuses the training images — with ~10 images a held-out split costs labels, says little."""
    task = project.task_type
    classes = list(project.classes)
    class_idx = {c.id: i for i, c in enumerate(classes)}
    names = {c.id: c.name for c in classes}
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if task != "classify":
        (out / "images").mkdir()
        (out / "labels").mkdir()
    for im in images:
        src = storage.image_path(project.id, im.filename)
        if not src.exists():
            continue
        name = f"{im.id}_{im.filename}"
        if task == "classify":
            tag = next(a for a in im.annotations if a.kind == "classification")
            for split in ("train", "val"):
                dst = out / split / class_folder(names[tag.class_id]) / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        else:
            shutil.copyfile(src, out / "images" / name)
            exporter._write_label_txt(out / "labels" / f"{Path(name).stem}.txt", exporter._ann_dicts(im),
                                      class_idx, im.width, im.height, task)
    if task == "classify":
        return out
    data = {"path": str(out.resolve()), "train": "images", "val": "images",
            "names": {i: c.name for i, c in enumerate(classes)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return out / "data.yaml"


# ── worker side (train_worker.py, after/while training) ──────────────
def track_epochs(model, db, run_id: int):
    """Mirror training progress into the run row, which the Auto-label page polls."""
    run = db.get(BootstrapRun, run_id)

    def on_epoch_end(tr):
        run.epoch = tr.epoch + 1
        run.message = f"Training — epoch {tr.epoch + 1}/{tr.epochs}"
        db.commit()

    model.add_callback("on_train_epoch_end", on_epoch_end)


def _save(db, iid: int, seen: str, preds: list[dict], class_map: dict) -> str:
    """Write one image's predictions, but only if nobody touched it since the round picked it."""
    claimed = jobs.claim_image(db, iid, seen)
    if not claimed or not preds:        # touched meanwhile / nothing found: an earlier guess stays
        db.rollback()
        return "touched" if not claimed else "empty"
    db.query(Annotation).filter(Annotation.image_id == iid, Annotation.source == "model") \
        .delete(synchronize_session=False)    # this round's guess replaces an unreviewed earlier one
    for p in preds:
        db.add(Annotation(image_id=iid, class_id=class_map[p["class_name"]], kind=p["kind"],
                          data=json.dumps(p["data"]), source="model", confidence=p["confidence"],
                          origin="bootstrap"))
    db.execute(update(ImageAsset).where(ImageAsset.id == iid).values(status="auto-annotated")
               .execution_options(synchronize_session=False))
    db.commit()
    return "labeled"


def label_phase(db, run_id: int, weight_id: int | None):
    """Label the round's targets with the freshly trained weights."""
    run = db.get(BootstrapRun, run_id)
    db.refresh(run)
    if run.status not in ACTIVE:
        return                                  # stopped meanwhile
    shutil.rmtree(_dataset_dir(run.train_job_id), ignore_errors=True)   # copies — not needed any more
    if not weight_id:
        raise BootstrapError("Training finished without a best.pt.")
    project = db.get(Project, run.project_id)
    cfg = json.loads(run.config or "{}")
    refresh_unreviewed = cfg.get("refresh_unreviewed", True)
    relabel_rejected = cfg.get("relabel_rejected", False)
    other_tools = cfg.get("relabel_other_tools", False)
    conf = cfg.get("conf", 0.4)
    valid = {c.id for c in project.classes}
    class_map = {k: v for k, v in (cfg.get("class_map") or {c.name: c.id for c in project.classes}).items()
                 if v in valid}
    targets = [(im.id, im.filename, im.status) for im in project.images
               if is_target(im, refresh_unreviewed, relabel_rejected, other_tools)]

    job = AutolabelJob(project_id=project.id, model_name=f"Bootstrap r{run.round}", total=len(targets))
    db.add(job)
    db.commit()
    run.status, run.weight_id, run.autolabel_job_id = "labeling", weight_id, job.id
    run.target_images = len(targets)
    run.message = f"Labeling {len(targets)} image(s)…"
    db.commit()

    labeled, skipped, log, errors = [], [], [], []
    for n, (iid, filename, seen) in enumerate(targets, 1):
        line = f"[{n}/{job.total}] {filename}"
        path = storage.image_path(project.id, filename)
        try:
            if not path.exists():
                raise inference.InferenceError("file missing")
            preds = inference.predict(f"weight:{weight_id}", str(path), project.task_type, conf,
                                      cfg.get("iou", 0.5), cfg.get("device", "auto"), db)
            # classification always returns its top class — the threshold applies here
            preds = [p for p in preds if p["confidence"] >= conf and p["class_name"] in class_map]
            kept, dropped = jobs.drop_duplicates(        # one box per object
                [jobs.shape(p["class_name"], p["kind"], p["data"], conf=p["confidence"], p=p) for p in preds],
                [], cfg.get("max_overlap", 0.7))
            preds = [k["p"] for k in kept]
            outcome = _save(db, iid, seen, preds, class_map)
        except Exception as e:                  # one unreadable image must not end the round
            db.rollback()
            skipped.append(filename)
            errors.append(str(e))
            log.append(f"{line} — failed: {e}")
        else:
            if outcome == "labeled":
                labeled.append(iid)
                job.images_labeled += 1
                job.annotations_added += len(preds)
                log.append(f"{line} — {len(preds)} object(s) found"
                           + (f", {dropped} duplicate(s) skipped" if dropped else "")
                           + (" (was rejected)" if seen == "needs_reannotation" else ""))
            elif outcome == "touched":
                log.append(f"{line} — labeled or reviewed by hand meanwhile, left alone")
            else:
                log.append(f"{line} — nothing detected" +
                           (" (kept the earlier prediction)" if seen != "unannotated" else ""))
        job.processed = n
        job.skipped = json.dumps(skipped)
        job.labeled_image_ids = json.dumps(labeled)
        job.log = "\n".join(log[-300:])
        db.commit()

    if targets and len(skipped) == len(targets):
        raise BootstrapError(f"Every image failed — first error: {errors[0]}")
    job.status = "done"
    db.refresh(run)
    if run.status in ACTIVE:
        run.status, run.finished_at = "done", _now()
        run.message = (f"Labeled {job.images_labeled} of {job.total} image(s) · "
                       f"+{job.annotations_added} annotations")
    db.commit()


def fail(db, run_id: int, error: str):
    """Worker-side failure — leaves a run the user already stopped alone."""
    db.rollback()
    run = db.get(BootstrapRun, run_id)
    if run is None:
        return
    db.refresh(run)
    shutil.rmtree(_dataset_dir(run.train_job_id), ignore_errors=True)
    if run.status in ACTIVE:
        run.status, run.error, run.finished_at = "failed", error, _now()
        _close_autolabel(db, run, error)
        db.commit()


def _close_autolabel(db, run: BootstrapRun, error: str):
    aj = db.get(AutolabelJob, run.autolabel_job_id) if run.autolabel_job_id else None
    if aj and aj.status == "running":
        aj.status, aj.error = "failed", error


# ── server side ──────────────────────────────────────────────────────
def _train_job(db, run: BootstrapRun) -> TrainJob | None:
    """The run's TrainJob — None if it was deleted (or its id reused by another job)."""
    job = db.get(TrainJob, run.train_job_id) if run.train_job_id else None
    return job if job and trainer.job_params(job).get("bootstrap_run") == run.id else None


def _worker_running(run: BootstrapRun, job: TrainJob | None) -> bool:
    if job is None:
        return False
    if not job.pid:                     # still building the training set, worker not started yet
        return job.status == "queued" and (_now() - run.created_at).total_seconds() < LAUNCH_GRACE
    return trainer.worker_alive(job)


def refresh(db, run: BootstrapRun) -> BootstrapRun:
    """A worker that died without saying so (killed, out of memory, container restart)
    fails its run."""
    if run.status in ACTIVE:
        job = _train_job(db, run)
        if not _worker_running(run, job):
            db.refresh(run)
            if run.status in ACTIVE:
                run.status, run.finished_at = "failed", _now()
                run.error = "The worker stopped unexpectedly — see the training log on the Train page."
                _close_autolabel(db, run, run.error)
                if job and job.status in ("queued", "running"):
                    job.status = "failed"
                db.commit()
    return run


def stop(db, run: BootstrapRun) -> BootstrapRun:
    job = _train_job(db, run)
    if job and run.status in ACTIVE:
        trainer.kill_worker(job)        # the same worker trains and then labels
    if job and job.status in ("queued", "running"):
        job.status = "stopped"
    if run.status in ACTIVE:
        run.status, run.finished_at, run.message = "stopped", _now(), "Stopped."
        _close_autolabel(db, run, "Stopped by user")
    db.commit()
    return run
