"""The "Test model" feature: read-only inference over a dataset version's
test-split images, for visualization only — nothing gets saved.

inference.predict() is monkeypatched (same convention as test_autolabel.py):
this suite deliberately doesn't depend on a real model being loaded.
"""
from conftest import make_png_bytes


def _fake_predict(model_name, path, task_type, conf, iou, device, db):
    return [{"class_name": "cat", "kind": "bbox",
             "data": {"x": 5, "y": 5, "w": 10, "h": 10}, "confidence": 0.9}]


def _project_with_images(client, n=3, splits=None):
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    image_ids = []
    for i in range(n):
        files = [("files", (f"img{i}.png", make_png_bytes(100, 100), "image/png"))]
        iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]
        # exporter.generate() only considers annotated + approved images before
        # splitting — mirror that here so the version actually has a test split
        client.put(f"/api/projects/{pid}/images/{iid}/annotations",
                   json={"annotations": [{"class_id": cid, "kind": "bbox",
                                           "data": {"x": 10, "y": 10, "w": 20, "h": 20}}]})
        client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
        image_ids.append(iid)
    cfg = {"splits": splits} if splits else {}
    vid = client.post(f"/api/projects/{pid}/versions", json={"name": "v1", **cfg}).json()["id"]
    return pid, cid, image_ids, vid


def test_test_model_returns_predictions_for_the_test_split(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", _fake_predict)

    pid, cid, image_ids, vid = _project_with_images(client, n=3, splits={"train": 0, "val": 0, "test": 1.0})

    r = client.post(f"/api/projects/{pid}/test-model",
                    json={"version_id": vid, "model_name": "yolo11n.pt"})
    assert r.status_code == 200
    body = r.json()
    assert body["test_split_total"] == 3
    assert body["shown"] == 3
    assert body["skipped"] == []
    assert {im["id"] for im in body["images"]} == set(image_ids)
    for im in body["images"]:
        assert len(im["predictions"]) == 1
        pred = im["predictions"][0]
        assert pred["class_id"] == cid
        assert pred["source"] == "model"
        assert pred["color"]


def test_test_model_limit_and_offset_page_through_the_split(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", _fake_predict)

    pid, cid, image_ids, vid = _project_with_images(client, n=5, splits={"train": 0, "val": 0, "test": 1.0})

    r1 = client.post(f"/api/projects/{pid}/test-model",
                     json={"version_id": vid, "model_name": "yolo11n.pt", "limit": 2, "offset": 0})
    r2 = client.post(f"/api/projects/{pid}/test-model",
                     json={"version_id": vid, "model_name": "yolo11n.pt", "limit": 2, "offset": 2})
    assert r1.json()["shown"] == 2
    assert r2.json()["shown"] == 2
    ids1 = {im["id"] for im in r1.json()["images"]}
    ids2 = {im["id"] for im in r2.json()["images"]}
    assert ids1.isdisjoint(ids2)   # no overlap between pages


def test_test_model_rejects_version_with_empty_test_split(client):
    pid, cid, image_ids, vid = _project_with_images(client, n=3, splits={"train": 1.0, "val": 0, "test": 0})
    r = client.post(f"/api/projects/{pid}/test-model",
                    json={"version_id": vid, "model_name": "yolo11n.pt"})
    assert r.status_code == 400


def test_test_model_rejects_unresolvable_stored_weight(client):
    pid, cid, image_ids, vid = _project_with_images(client, n=3, splits={"train": 0, "val": 0, "test": 1.0})
    r = client.post(f"/api/projects/{pid}/test-model",
                    json={"version_id": vid, "model_name": "weight:99999"})
    assert r.status_code == 400


def test_test_model_rejects_unknown_version(client):
    pid, cid, image_ids, vid = _project_with_images(client, n=1, splits={"train": 0, "val": 0, "test": 1.0})
    r = client.post(f"/api/projects/{pid}/test-model",
                    json={"version_id": vid + 999, "model_name": "yolo11n.pt"})
    assert r.status_code == 404


def test_test_model_skips_images_on_inference_error(client, monkeypatch):
    from backend.routers import models as models_router
    from backend.services import inference as inference_module

    def _boom(*a, **k):
        raise inference_module.InferenceError("model failed to load")
    monkeypatch.setattr(models_router.inference, "predict", _boom)

    pid, cid, image_ids, vid = _project_with_images(client, n=2, splits={"train": 0, "val": 0, "test": 1.0})
    r = client.post(f"/api/projects/{pid}/test-model",
                    json={"version_id": vid, "model_name": "yolo11n.pt"})
    assert r.status_code == 200
    body = r.json()
    assert body["shown"] == 0
    assert len(body["skipped"]) == 2


def test_arbitrary_model_paths_are_refused_everywhere(client):
    """.pt files are pickles — a client-chosen path must never reach the loader."""
    pid, cid, image_ids, vid = _project_with_images(client, n=1, splits={"train": 0, "val": 0, "test": 1.0})
    for bad in ("/etc/evil.pt", "../../data/models/x.pt", "weight:abc"):
        assert client.post(f"/api/projects/{pid}/test-model",
                           json={"version_id": vid, "model_name": bad}).status_code == 400
        assert client.post(f"/api/projects/{pid}/autolabel",
                           json={"model_name": bad}).status_code == 400
        assert client.post(f"/api/projects/{pid}/train",
                           json={"version_id": vid, "model_arch": bad}).status_code == 400
    r = client.post(f"/api/projects/{pid}/images/{image_ids[0]}/sam",
                    json={"model_name": "/etc/evil.pt", "prompt": {"type": "point", "points": [[5, 5]], "labels": [1]}})
    assert r.status_code == 422
