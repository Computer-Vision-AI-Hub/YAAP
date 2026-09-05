"""Training job plot serving + delete-also-wipes-disk behavior.

Starting a real training job needs torch/ultralytics actually loaded, which
this suite deliberately doesn't depend on (see README's Testing section) —
so jobs are seeded directly in the DB, matching the pattern in
test_system.py's TrainJob tests.
"""
from backend.config import RUNS_DIR
from backend.db import ModelWeight, SessionLocal, TrainJob


def _seed_job(pid, status="done", with_weights=True):
    db = SessionLocal()
    try:
        job = TrainJob(project_id=pid, version_id=1, model_arch="yolo11n.pt", status=status)
        db.add(job)
        db.commit()
        run_dir = RUNS_DIR / f"job_{job.id}"
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        weights_path = str(run_dir / "weights" / "best.pt")
        if with_weights:
            (run_dir / "weights" / "best.pt").write_bytes(b"fake weights")
            job.weights_path = weights_path
            db.add(ModelWeight(project_id=pid, name="job weight", arch="yolo11n.pt",
                               task="detect", path=weights_path, source="trained"))
        db.commit()
        jid = job.id
    finally:
        db.close()
    return jid, run_dir, weights_path


def test_job_plot_404_for_unknown_name(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    jid, _, _ = _seed_job(pid)
    r = client.get(f"/api/projects/{pid}/train/{jid}/plot/not_a_real_plot")
    assert r.status_code == 404


def test_job_plot_404_before_it_exists(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    jid, _, _ = _seed_job(pid, status="running", with_weights=False)
    r = client.get(f"/api/projects/{pid}/train/{jid}/plot/results")
    assert r.status_code == 404


def test_job_plot_serves_file_once_it_exists(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    jid, run_dir, _ = _seed_job(pid)
    (run_dir / "results.png").write_bytes(b"\x89PNG\r\n fake png bytes")
    (run_dir / "confusion_matrix.png").write_bytes(b"\x89PNG\r\n other fake png bytes")

    r = client.get(f"/api/projects/{pid}/train/{jid}/plot/results")
    assert r.status_code == 200
    assert r.content == b"\x89PNG\r\n fake png bytes"

    r = client.get(f"/api/projects/{pid}/train/{jid}/plot/confusion_matrix")
    assert r.status_code == 200
    assert r.content == b"\x89PNG\r\n other fake png bytes"


def test_delete_job_wipes_run_dir_and_registered_weight(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    jid, run_dir, weights_path = _seed_job(pid)
    assert run_dir.exists()
    assert (run_dir / "weights" / "best.pt").exists()

    db_weights_before = client.get(f"/api/models?pid={pid}&task=detect").json()["weights"]
    assert any(w["name"] == "job weight" for w in db_weights_before)

    r = client.delete(f"/api/projects/{pid}/train/{jid}")
    assert r.status_code == 200

    assert not run_dir.exists()   # the whole job_<id> folder — weights, results.png, all of it
    db_weights_after = client.get(f"/api/models?pid={pid}&task=detect").json()["weights"]
    assert not any(w["name"] == "job weight" for w in db_weights_after)
    assert client.get(f"/api/projects/{pid}/train").json() == []


def test_delete_job_without_weights_still_removes_run_dir(client):
    """A job that never produced a best.pt (failed early) shouldn't error
    just because there's no ModelWeight to clean up."""
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    jid, run_dir, _ = _seed_job(pid, status="failed", with_weights=False)
    assert run_dir.exists()
    r = client.delete(f"/api/projects/{pid}/train/{jid}")
    assert r.status_code == 200
    assert not run_dir.exists()
