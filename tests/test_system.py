"""Auto-shutdown signal endpoints (see watch-idle.sh) and the guard that
must refuse to arm while a training job is active."""
from backend.db import SessionLocal, TrainJob
from backend.routers.models import LEAVE_MARKER


def test_leaving_writes_marker_when_idle(client):
    LEAVE_MARKER.unlink(missing_ok=True)
    r = client.post("/api/system/leaving")
    assert r.json() == {"ok": True}
    assert LEAVE_MARKER.exists()


def test_here_clears_marker(client):
    LEAVE_MARKER.write_text("123")
    r = client.post("/api/system/here")
    assert r.json() == {"ok": True}
    assert not LEAVE_MARKER.exists()


def test_leaving_refuses_while_job_running(client):
    LEAVE_MARKER.unlink(missing_ok=True)
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    vid_row = None
    db = SessionLocal()
    try:
        job = TrainJob(project_id=pid, version_id=1, model_arch="yolo11n.pt", status="running")
        db.add(job)
        db.commit()
    finally:
        db.close()

    r = client.post("/api/system/leaving")
    assert r.json()["ok"] is False
    assert not LEAVE_MARKER.exists()


def test_leaving_allowed_once_job_finishes(client):
    LEAVE_MARKER.unlink(missing_ok=True)
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    db = SessionLocal()
    try:
        job = TrainJob(project_id=pid, version_id=1, model_arch="yolo11n.pt", status="done")
        db.add(job)
        db.commit()
    finally:
        db.close()

    r = client.post("/api/system/leaving")
    assert r.json() == {"ok": True}
    assert LEAVE_MARKER.exists()
