"""Project + class CRUD."""
from conftest import make_png_bytes


def test_empty_status_counts_toward_annotated_count(client):
    """A "Mark empty" image has zero real annotation rows but should still
    count as resolved/labeled, not pending — see routers/projects.py."""
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    assert client.get(f"/api/projects/{pid}").json()["annotated_count"] == 0
    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "empty"})
    assert client.get(f"/api/projects/{pid}").json()["annotated_count"] == 1


def test_create_and_list_project(client):
    r = client.post("/api/projects", json={"name": "Cats", "task_type": "detect"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Cats"
    assert body["task_type"] == "detect"
    assert body["image_count"] == 0
    assert body["class_count"] == 0

    r = client.get("/api/projects")
    assert [p["name"] for p in r.json()] == ["Cats"]


def test_create_project_defaults_untitled_name(client):
    r = client.post("/api/projects", json={"name": "   "})
    assert r.status_code == 200
    assert r.json()["name"] == "Untitled"


def test_get_missing_project_404(client):
    r = client.get("/api/projects/999")
    assert r.status_code == 404


def test_delete_project(client):
    pid = client.post("/api/projects", json={"name": "Temp"}).json()["id"]
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_add_class_and_reject_duplicate(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]

    r = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"})
    assert r.status_code == 200
    assert r.json()["name"] == "cat"

    r = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"})
    assert r.status_code == 409


def test_add_class_rejects_empty_name(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/classes", json={"name": "   "})
    assert r.status_code == 400


def test_class_gets_next_palette_color_when_unset(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    c1 = client.post(f"/api/projects/{pid}/classes", json={"name": "a"}).json()
    c2 = client.post(f"/api/projects/{pid}/classes", json={"name": "b"}).json()
    assert c1["color"] != c2["color"]
    assert c1["order_idx"] == 0 and c2["order_idx"] == 1


def test_update_and_delete_class(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "a"}).json()["id"]

    r = client.patch(f"/api/projects/{pid}/classes/{cid}", json={"name": "b"})
    assert r.json()["name"] == "b"

    assert client.delete(f"/api/projects/{pid}/classes/{cid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}").json()["class_count"] == 0


def test_project_list_counts_and_constant_query_count(client):
    from sqlalchemy import event
    from backend.db import engine
    from conftest import make_png_bytes

    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    ids = []
    for i in range(6):
        files = [("files", (f"i{i}.png", make_png_bytes(20, 20), "image/png"))]
        ids.append(client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"])
    client.put(f"/api/projects/{pid}/images/{ids[0]}/annotations",
               json={"annotations": [{"class_id": cid, "kind": "bbox", "data": {"x": 1, "y": 1, "w": 5, "h": 5}}]})
    client.patch(f"/api/projects/{pid}/images/{ids[1]}", json={"status": "empty"})
    client.post(f"/api/projects/{pid}/versions", json={"name": "v1", "approved_only": False})

    def count_queries():
        n = [0]
        def on_exec(*a, **k): n[0] += 1
        event.listen(engine, "before_cursor_execute", on_exec)
        try:
            body = client.get("/api/projects").json()
        finally:
            event.remove(engine, "before_cursor_execute", on_exec)
        return n[0], body

    n_before, body = count_queries()
    p = next(x for x in body if x["id"] == pid)
    assert p["image_count"] == 6
    assert p["annotated_count"] == 2       # one annotated, one marked empty
    assert p["class_count"] == 1
    assert p["version_count"] == 1

    for i in range(10):                    # more unlabeled images must not add queries
        files = [("files", (f"x{i}.png", make_png_bytes(20, 20), "image/png"))]
        client.post(f"/api/projects/{pid}/images", files=files)
    n_after, _ = count_queries()
    assert n_after == n_before


def test_labeled_count_means_checked_by_a_person(client):
    """Model output nobody has reviewed and rejected images are not "labeled"."""
    from backend.db import ImageAsset, SessionLocal
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    statuses = ["annotated", "approved", "empty", "auto-annotated", "review", "needs_reannotation", "unannotated"]
    files = [("files", (f"{s}.png", make_png_bytes(), "image/png")) for s in statuses]
    ids = [im["id"] for im in client.post(f"/api/projects/{pid}/images", files=files).json()["created"]]
    db = SessionLocal()
    for iid, s in zip(ids, statuses):
        db.get(ImageAsset, iid).status = s
    db.commit()
    db.close()
    p = client.get(f"/api/projects/{pid}").json()
    assert (p["image_count"], p["annotated_count"], p["predicted_count"]) == (7, 3, 2)


def _fake_worker(job_id):
    """A harmless process whose command line looks like the training worker's."""
    import subprocess
    return subprocess.Popen(["sh", "-c", "sleep 60; true", "backend.services.train_worker", "--job", str(job_id)],
                            start_new_session=True)


def test_deleting_a_project_removes_everything_it_owns(client):
    from backend.config import MODELS_DIR, RUNS_DIR
    from backend.db import AutolabelJob, BootstrapRun, ModelWeight, SessionLocal, TrainJob

    def project_with_stuff(name):
        pid = client.post("/api/projects", json={"name": name}).json()["id"]
        files = [("files", ("a.png", make_png_bytes(), "image/png"))]
        client.post(f"/api/projects/{pid}/images", files=files)
        db = SessionLocal()
        job = TrainJob(project_id=pid, version_id=0, model_arch="yolo11n.pt", status="done")
        db.add(job)
        db.commit()
        run = RUNS_DIR / f"job_{job.id}"
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pt").write_bytes(b"w")
        log = RUNS_DIR / f"job_{job.id}.log"
        log.write_text("log")
        job.weights_path, job.log_path = str(run / "weights" / "best.pt"), str(log)
        upload = MODELS_DIR / f"{name}_upload.pt"
        upload.write_bytes(b"w")
        db.add_all([ModelWeight(project_id=pid, name="trained", path=job.weights_path, source="trained"),
                    ModelWeight(project_id=pid, name="uploaded", path=str(upload), source="uploaded"),
                    AutolabelJob(project_id=pid, model_name="m", status="done"),
                    BootstrapRun(project_id=pid, status="done")])
        db.commit()
        db.close()
        return pid, run, log, upload

    pid, run, log, upload = project_with_stuff("gone")
    keep_pid, keep_run, keep_log, keep_upload = project_with_stuff("kept")
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}

    assert not run.exists() and not log.exists() and not upload.exists()
    db = SessionLocal()
    for table in (TrainJob, ModelWeight, AutolabelJob, BootstrapRun):
        assert db.query(table).filter_by(project_id=pid).count() == 0
        assert db.query(table).filter_by(project_id=keep_pid).count() >= 1
    db.close()
    assert keep_run.exists() and keep_log.exists() and keep_upload.exists()
    keep_upload.unlink()


def test_deleting_a_project_stops_its_running_training(client):
    import signal
    from backend.db import SessionLocal, TrainJob
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    db = SessionLocal()
    job = TrainJob(project_id=pid, version_id=0, model_arch="yolo11n.pt", status="running")
    db.add(job)
    db.commit()
    proc = _fake_worker(job.id)
    job.pid = proc.pid
    db.commit()
    db.close()
    try:
        client.delete(f"/api/projects/{pid}")
        assert proc.wait(timeout=5) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()


def test_deleting_a_project_keeps_a_weight_file_another_project_uses(client):
    from backend.config import MODELS_DIR
    from backend.db import ModelWeight, SessionLocal
    a = client.post("/api/projects", json={"name": "A"}).json()["id"]
    b = client.post("/api/projects", json={"name": "B"}).json()["id"]
    shared = MODELS_DIR / "shared.pt"
    shared.write_bytes(b"w")
    db = SessionLocal()
    db.add_all([ModelWeight(project_id=a, name="w", path=str(shared)),
                ModelWeight(project_id=b, name="w", path=str(shared))])
    db.commit()
    db.close()
    client.delete(f"/api/projects/{a}")
    assert shared.exists()
    shared.unlink()
