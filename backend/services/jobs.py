"""Shared plumbing for background jobs: writing model output safely next to a user
who keeps editing, and the idle-shutdown handshake with watch-idle.sh."""
from __future__ import annotations

import time

from sqlalchemy import update

from ..config import DATA_DIR
from ..db import AutolabelJob, BootstrapRun, ImageAsset, SessionLocal, TrainJob

LEAVE_MARKER = DATA_DIR / ".leave_at"         # watch-idle.sh stops the stack 60 s after this
LEAVE_PENDING = DATA_DIR / ".leave_pending"   # tab closed while a job ran: leave once it ends


SAME_CLASS_IOU = 0.5    # two boxes of one class overlapping this much are one object (as in NMS)


def box_of(kind: str, data: dict) -> dict | None:
    """The pixel box an annotation covers — polygons by their bounding box; None for a
    whole-image (classification) label."""
    if kind == "bbox":
        return data
    pts = data.get("points") or data.get("polygon")
    if kind == "polygon" and pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}
    return None


def shape(cls, kind: str, data: dict, **extra) -> dict:
    """An item for drop_duplicates: its class, box and — for polygons — the outline."""
    pts = (data.get("points") or data.get("polygon")) if kind == "polygon" else None
    return {"cls": cls, "box": box_of(kind, data), "poly": pts, **extra}


def poly_iou(p: list, q: list) -> float:
    """Overlap of two outlines by area: both filled into a small common raster. Two triangles
    that split one square have the same box (box IoU 1) but no shared area (polygon IoU 0)."""
    import cv2
    import numpy as np
    pts = np.asarray(list(p) + list(q), dtype=np.float32)
    lo, hi = pts.min(0), pts.max(0)
    scale = 256.0 / max(float((hi - lo).max()), 1.0)
    w, h = (np.ceil((hi - lo) * scale) + 3).astype(int)
    masks = []
    for poly in (p, q):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round((np.asarray(poly, np.float32) - lo) * scale).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    union = np.count_nonzero(masks[0] | masks[1])
    return np.count_nonzero(masks[0] & masks[1]) / union if union else 0.0


def overlap(a: dict, b: dict) -> float:
    """IoU of two drop_duplicates items: by area when both are polygons, else by box."""
    o = iou(a["box"], b["box"])
    if o > 0 and a.get("poly") and b.get("poly"):
        return poly_iou(a["poly"], b["poly"])
    return o


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    union = a["w"] * a["h"] + b["w"] * b["h"] - ix * iy
    return ix * iy / union if union > 0 else 0.0


def drop_duplicates(new: list[dict], existing: list[dict], max_overlap: float = 0.7) -> tuple[list[dict], int]:
    """One annotation per object. `new` items ({"cls", "box", "conf", …}) are tried best-first
    against everything kept so far and against `existing` ({"cls", "box"}), which always wins.
    Returns (kept items, how many were dropped)."""
    ranked = sorted(new, key=lambda d: -d["conf"])
    if max_overlap >= 1:
        return ranked, 0
    taken, kept = list(existing), []
    for n in ranked:
        if n["box"] is None:
            dup = any(t["box"] is None for t in taken)
        else:
            dup = any(t["box"] is not None and (o := overlap(n, t)) > 0 and
                      (o >= max_overlap or (o >= SAME_CLASS_IOU and t["cls"] == n["cls"])) for t in taken)
        if not dup:
            kept.append(n)
            taken.append(n)
    return kept, len(ranked) - len(kept)


def claim_image(db, iid: int, seen: str) -> bool:
    """True if the image still has the status the job saw when it picked it. The no-op
    UPDATE takes SQLite's write lock, so this check and the job's writes that follow
    (until commit) are one atomic step against the editor and Review."""
    return db.execute(update(ImageAsset).where(ImageAsset.id == iid, ImageAsset.status == seen)
                      .values(status=seen).execution_options(synchronize_session=False)).rowcount == 1


# ── idle shutdown ────────────────────────────────────────────────────
# Open tabs announce themselves (POST /system/here) on load and every ~20 s; a tab that closes
# says so (/system/leaving). The stack is only stopped when no other tab has been seen recently —
# closing one of two tabs used to shut YAAP down under the user's feet.
TAB_TIMEOUT = 60.0                    # seconds without a heartbeat after which a tab counts as gone
_tabs: dict[str, float] = {}          # tab id → last heartbeat (time.monotonic)


def tab_seen(tab: str | None):
    if tab:
        _tabs[tab[:64]] = time.monotonic()


def other_tabs(tab: str | None) -> int:
    now = time.monotonic()
    for t, seen in list(_tabs.items()):
        if now - seen > TAB_TIMEOUT:
            _tabs.pop(t, None)
    return sum(1 for t in _tabs if t != tab)


def busy_reason(db) -> str | None:
    from . import trainer
    for job in db.query(TrainJob).filter(TrainJob.status.in_(["queued", "running"])).all():
        if trainer.refresh_status(db, job).status in ("queued", "running"):   # a dead worker isn't busy
            return "training job active"
    from . import bootstrap
    for run in db.query(BootstrapRun).filter(BootstrapRun.status.in_(bootstrap.ACTIVE)).all():
        if bootstrap.refresh(db, run).status in bootstrap.ACTIVE:     # its labeling outlives the TrainJob
            return "bootstrap running"
    if db.query(AutolabelJob).filter(AutolabelJob.status == "running").first():
        return "auto-label job running"
    return None


def request_leave(db, tab: str | None = None) -> dict:
    """The tab was closed. Another tab still open → nothing; idle → arm the shutdown now;
    busy → arm it when the job ends."""
    if tab:
        _tabs.pop(tab[:64], None)
    if other_tabs(tab):
        return {"ok": False, "reason": "another YAAP tab is open"}
    reason = busy_reason(db)
    if reason:
        LEAVE_PENDING.write_text(str(int(time.time())))
        return {"ok": False, "reason": reason}
    LEAVE_PENDING.unlink(missing_ok=True)
    LEAVE_MARKER.write_text(str(int(time.time())))
    return {"ok": True}


def cancel_leave(tab: str | None = None):
    tab_seen(tab)
    LEAVE_PENDING.unlink(missing_ok=True)
    LEAVE_MARKER.unlink(missing_ok=True)


def job_finished(db=None):
    """Call when a background job ends: a shutdown deferred by it can go ahead now."""
    if not LEAVE_PENDING.exists():
        return
    own = db is None
    db = db or SessionLocal()
    try:
        if busy_reason(db) is None:
            LEAVE_PENDING.unlink(missing_ok=True)
            LEAVE_MARKER.write_text(str(int(time.time())))
    finally:
        if own:
            db.close()


def fail_interrupted_jobs():
    """At startup: in-process jobs (auto-label, find-all, ranking) died with the old
    server — without this they would show "running" forever. Bootstrap's labeling runs
    in its own worker and is checked by bootstrap.refresh() instead."""
    db = SessionLocal()
    try:
        owned = {r.autolabel_job_id for r in db.query(BootstrapRun).all() if r.autolabel_job_id}
        for job in db.query(AutolabelJob).filter(AutolabelJob.status == "running").all():
            if job.id not in owned:
                job.status, job.error = "failed", "Interrupted — the server stopped while this job ran."
        db.commit()
        # a shutdown deferred until "the job ends" belonged to a job that just died with the old
        # server: drop it, or it fired at the end of some later, unrelated job
        LEAVE_PENDING.unlink(missing_ok=True)
    finally:
        db.close()
