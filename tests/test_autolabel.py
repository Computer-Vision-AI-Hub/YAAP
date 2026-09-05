"""Auto-label as a background job: progress tracking, and the new
preview/approve gate — predictions must land as status="annotated", never
"review" directly; only an explicit approval (per-image or bulk, both just
the ordinary PATCH the frontend calls) promotes an image to "review".

inference.predict() is monkeypatched everywhere here: it needs a real model
loaded (torch/ultralytics), which this suite deliberately doesn't depend on
— see README's Testing section. Everything around that one call is real.
"""
from conftest import make_png_bytes


def _fake_predict(model_name, path, task_type, conf, iou, device, db):
    return [{"class_name": "cat", "kind": "bbox",
             "data": {"x": 5, "y": 5, "w": 10, "h": 10}, "confidence": 0.9}]


def _project_with_image(client):
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(100, 100), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]
    return pid, iid


def test_autolabel_runs_as_job_and_leaves_status_auto_annotated(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", _fake_predict)

    pid, iid = _project_with_image(client)
    r = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"})
    assert r.status_code == 200
    jid = r.json()["id"]

    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert job["status"] == "done"
    assert job["images_labeled"] == 1
    assert job["annotations_added"] == 1
    assert job["processed"] == job["total"] == 1
    assert "object(s) found" in job["log"]

    im = client.get(f"/api/projects/{pid}/images").json()[0]
    # neither "annotated" (would hide it from a dedicated auto-label triage
    # filter) nor "review" (would skip the approval gate entirely)
    assert im["status"] == "auto-annotated"


def test_clearing_annotations_resets_status_to_unannotated(client, monkeypatch):
    """The QC grid's erase button (🗑) is just this endpoint with an empty
    list — works from any prior status, including auto-annotated."""
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", _fake_predict)

    pid, iid = _project_with_image(client)
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"})
    assert client.get(f"/api/projects/{pid}/images").json()[0]["status"] == "auto-annotated"

    r = client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": []})
    assert r.json()["status"] == "unannotated"
    assert r.json()["annotations"] == []
    assert client.get(f"/api/projects/{pid}/images/{iid}/annotations").json() == []


def test_autolabel_preview_lists_only_this_jobs_images(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", _fake_predict)
    pid, iid = _project_with_image(client)
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"}).json()["id"]

    r = client.get(f"/api/projects/{pid}/autolabel/{jid}/preview")
    assert r.status_code == 200
    body = r.json()
    assert body["job"]["id"] == jid
    assert len(body["images"]) == 1
    assert body["images"][0]["id"] == iid
    assert len(body["images"][0]["annotations"]) == 1
    assert body["images"][0]["annotations"][0]["source"] == "model"


def test_approving_preview_image_moves_it_to_review(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", _fake_predict)
    pid, iid = _project_with_image(client)
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"})

    r = client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "review"})
    assert r.json()["status"] == "review"


def test_autolabel_with_no_images_returns_400(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"})
    assert r.status_code == 400


def test_inference_error_is_logged_not_fatal(client, monkeypatch):
    from backend.routers import models as models_router
    from backend.services import inference as inference_module

    def _boom(*a, **k):
        raise inference_module.InferenceError("model failed to load")
    monkeypatch.setattr(models_router.inference, "predict", _boom)

    pid, iid = _project_with_image(client)
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"}).json()["id"]

    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert job["status"] == "done"          # a bad image doesn't crash the whole batch
    assert job["images_labeled"] == 0
    assert "a.png" in job["skipped"]
    assert "inference failed" in job["log"]
