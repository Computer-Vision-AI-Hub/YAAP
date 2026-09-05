"""SAM point-prompt shape — the bug this guards against: a flat points/labels
list gets reshaped by ultralytics' SAM predictor into N separate single-point
objects (each its own mask) instead of one mask jointly refined by all N
clicks. Confirmed against the real loaded sam3.pt before fixing (see
services/sam.py's predict()): flat shape returned 2 masks for 2 clicks on
distinct features; nested shape correctly returned 1.

The model itself is mocked here — this only checks the shape `predict()`
hands to it, not SAM's actual segmentation output.
"""
from conftest import make_png_bytes


class _FakeResult:
    masks = None
    boxes = None


def test_point_prompt_nests_points_and_labels_as_one_object(client, monkeypatch):
    from backend.services import sam as sam_module

    captured = {}

    def fake_model(image_path, **kwargs):
        captured.update(kwargs)
        return [_FakeResult()]

    monkeypatch.setattr(sam_module, "_load_sam", lambda name: fake_model)

    pid = client.post("/api/projects", json={"name": "P", "task_type": "segment"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(100, 100), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    body = {"prompt": {"type": "point", "points": [[10, 10], [20, 20], [30, 30]],
                       "labels": [1, 1, 0]}}
    r = client.post(f"/api/projects/{pid}/images/{iid}/sam", json=body)
    assert r.status_code == 200

    # (1, N, 2) / (1, N) — ONE object carrying all 3 points, not 3 separate objects
    assert captured["points"] == [[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]]
    assert captured["labels"] == [[1, 1, 0]]


def test_single_point_prompt_still_nested_consistently(client, monkeypatch):
    from backend.services import sam as sam_module

    captured = {}
    monkeypatch.setattr(sam_module, "_load_sam",
                        lambda name: (lambda image_path, **kw: captured.update(kw) or [_FakeResult()]))

    pid = client.post("/api/projects", json={"name": "P", "task_type": "segment"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(100, 100), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    body = {"prompt": {"type": "point", "points": [[42, 42]], "labels": [1]}}
    client.post(f"/api/projects/{pid}/images/{iid}/sam", json=body)

    assert captured["points"] == [[[42.0, 42.0]]]
    assert captured["labels"] == [[1]]
