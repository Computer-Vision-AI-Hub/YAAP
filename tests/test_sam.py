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


def test_clicks_on_one_image_reuse_its_encoding(client, monkeypatch):
    """Encoding a photo costs ~2 s, a prompt on the encoded features ~0.03 s: only the first click on an image
    may encode it, and another image must never get the previous image's features."""
    from backend.services import sam as sam_module

    class FakePredictor:
        def __init__(self):
            self.encoded, self.current, self.prompts = [], None, []

        def set_image(self, path):
            self.encoded.append(path); self.current = path

        def reset_image(self):
            self.current = None

        def __call__(self, **kw):
            assert self.current is not None, "prompted without an encoded image"
            self.prompts.append((self.current, kw))
            return [_FakeResult()]

    class FakeModel:
        def __init__(self):
            self.predictor = FakePredictor()

        def __call__(self, *a, **k):
            raise AssertionError("a full call would re-encode the image")

        def to(self, device):
            return self

    fake = FakeModel()
    monkeypatch.setattr(sam_module, "_load_sam", lambda name: fake)
    sam_module._encoded["key"] = None
    pid = client.post("/api/projects", json={"name": "P", "task_type": "segment"}).json()["id"]
    ids = [client.post(f"/api/projects/{pid}/images", files=[("files", (f"{n}.png", make_png_bytes(64, 64), "image/png"))])
           .json()["created"][0]["id"] for n in "ab"]
    box = {"prompt": {"type": "box", "box": [1, 1, 20, 20]}}
    for iid in (ids[0], ids[0], ids[0], ids[1], ids[0]):
        assert client.post(f"/api/projects/{pid}/images/{iid}/sam", json=box).status_code == 200
    names = [p.rsplit("/", 1)[-1] for p in fake.predictor.encoded]
    assert names == ["a.png", "b.png", "a.png"]                  # 5 clicks, 3 encodings
    assert [p.rsplit("/", 1)[-1] for p, _ in fake.predictor.prompts] == ["a.png", "a.png", "a.png", "b.png", "a.png"]
    assert fake.predictor.prompts[0][1] == {"bboxes": [[1.0, 1.0, 20.0, 20.0]]}
