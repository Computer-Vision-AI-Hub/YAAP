"""Version health stats, review ordering data (min_conf), SAM find-all and
active-learning scoring. Models are mocked — only the plumbing is checked."""
import json

import yaml
from conftest import make_png_bytes

from backend.db import DatasetVersion, SessionLocal
from backend.services import exporter, health


def _project(client, task="detect", n_images=1):
    pid = client.post("/api/projects", json={"name": "P", "task_type": task}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "port"}).json()["id"]
    ids = []
    for i in range(n_images):
        files = [("files", (f"a{i}.png", make_png_bytes(100, 100), "image/png"))]
        ids.append(client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"])
    return pid, cid, ids


# ── health ───────────────────────────────────────────────────────────
def _fake_version(pid):
    db = SessionLocal()
    v = DatasetVersion(project_id=pid, name="v1", status="ready", stats="{}")
    db.add(v); db.commit()
    vid = v.id
    db.close()
    vdir = exporter.version_dir(pid, vid)
    for split, n in (("train", 4), ("val", 1)):
        (vdir / split / "images").mkdir(parents=True)
        (vdir / split / "labels").mkdir(parents=True)
        for i in range(n):
            (vdir / split / "images" / f"{split}{i}.jpg").write_bytes(make_png_bytes(8, 8, (i * 40, 0, 0)))
            lines = "" if i == 0 else "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.05 0.05\n"
            (vdir / split / "labels" / f"{split}{i}.txt").write_text(lines)
    (vdir / "data.yaml").write_text(yaml.safe_dump({"names": {0: "cat", 1: "dog"}}))
    return vid


def test_health_counts_classes_splits_and_warns(client):
    pid, _, _ = _project(client)
    vid = _fake_version(pid)
    h = client.get(f"/api/projects/{pid}/versions/{vid}/health").json()
    assert h["images"] == {"train": 4, "val": 1, "test": 0}
    assert h["empty_images"]["train"] == 1
    by = {c["name"]: c for c in h["classes"]}
    assert by["cat"]["train"] == 3 and by["cat"]["val"] == 0 and by["dog"]["total"] == 3
    msgs = " ".join(w["msg"] for w in h["warnings"])
    assert "only" in msgs and "no labels in the val split" in msgs
    assert 0 <= h["score"] < 100
    # cached into the version's stats
    db = SessionLocal()
    assert "health" in json.loads(db.get(DatasetVersion, vid).stats)
    db.close()


def test_health_flags_cross_split_duplicates(client, tmp_path):
    for s in ("train", "val"):
        (tmp_path / s / "images").mkdir(parents=True)
        (tmp_path / s / "labels").mkdir(parents=True)
        (tmp_path / s / "images" / "x.jpg").write_bytes(make_png_bytes(8, 8))
        (tmp_path / s / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["a"]}))
    h = health.compute(tmp_path, "detect")
    assert h["duplicates"]["cross_split"] == 1
    assert any(w["level"] == "error" and "leakage" in w["msg"] for w in h["warnings"])


def test_health_rejects_unready_version(client):
    pid, _, _ = _project(client)
    db = SessionLocal()
    v = DatasetVersion(project_id=pid, name="v", status="building"); db.add(v); db.commit()
    vid = v.id; db.close()
    assert client.get(f"/api/projects/{pid}/versions/{vid}/health").status_code == 400


# ── review ordering data ─────────────────────────────────────────────
def test_image_list_exposes_min_model_confidence(client):
    pid, cid, (iid,) = _project(client)
    anns = [{"class_id": cid, "kind": "bbox", "data": {"x": 1, "y": 1, "w": 5, "h": 5}, "source": "model", "confidence": 0.9},
            {"class_id": cid, "kind": "bbox", "data": {"x": 9, "y": 9, "w": 5, "h": 5}, "source": "model", "confidence": 0.35},
            {"class_id": cid, "kind": "bbox", "data": {"x": 20, "y": 9, "w": 5, "h": 5}, "source": "manual", "confidence": 1.0}]
    client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": anns})
    assert client.get(f"/api/projects/{pid}/images").json()[0]["min_conf"] == 0.35


def test_min_conf_is_null_for_manual_only_images(client):
    pid, cid, (iid,) = _project(client)
    client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": [
        {"class_id": cid, "kind": "bbox", "data": {"x": 1, "y": 1, "w": 5, "h": 5}}]})
    assert client.get(f"/api/projects/{pid}/images").json()[0]["min_conf"] is None


# ── SAM exemplar + dataset-wide find-all ─────────────────────────────
_MATCHES = [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
             "score": 0.8, "label": ""},
            {"polygon": [[20, 20], [40, 20], [40, 40], [20, 40]], "bbox": {"x": 20, "y": 20, "w": 20, "h": 20},
             "score": 0.6, "label": ""}]


def test_exemplar_prompt_returns_all_matches(client, monkeypatch):
    from backend.services import sam
    seen = {}
    monkeypatch.setattr(sam, "predict_all", lambda path, model, box=None, **kw: seen.update(box=box, **kw) or _MATCHES)
    pid, _, (iid,) = _project(client)
    r = client.post(f"/api/projects/{pid}/images/{iid}/sam",
                    json={"prompt": {"type": "exemplar", "box": [1, 2, 30, 40]}, "conf": 0.3})
    assert r.status_code == 200 and len(r.json()["results"]) == 2
    assert seen["box"] == [1, 2, 30, 40] and seen["conf"] == 0.3


def test_find_all_job_saves_model_annotations_for_review(client, monkeypatch):
    from backend.services import sam
    monkeypatch.setattr(sam, "predict_phrases", lambda path, model, phrases, **k: {p: _MATCHES for p in phrases})
    pid, cid, (i1, i2) = _project(client, n_images=2)
    client.put(f"/api/projects/{pid}/images/{i2}/annotations", json={"annotations": [
        {"class_id": cid, "kind": "bbox", "data": {"x": 1, "y": 1, "w": 5, "h": 5}}]})
    r = client.post(f"/api/projects/{pid}/sam/find-all", json={"class_id": cid, "text": "port"})
    assert r.status_code == 200
    job = client.get(f"/api/projects/{pid}/autolabel/{r.json()['id']}").json()
    assert job["status"] == "done" and job["total"] == 1       # i2 already annotated → skipped
    assert job["annotations_added"] == 2
    ims = {i["id"]: i for i in client.get(f"/api/projects/{pid}/images").json()}
    assert ims[i1]["status"] == "auto-annotated" and ims[i1]["min_conf"] == 0.6
    assert ims[i2]["annotation_count"] == 1


def test_find_all_refused_for_classification(client):
    pid, cid, _ = _project(client, task="classify")
    r = client.post(f"/api/projects/{pid}/sam/find-all", json={"class_id": cid, "text": "x"})
    assert r.status_code == 400


# ── active learning ──────────────────────────────────────────────────
def test_uncertainty_is_one_minus_mean_confidence():
    from backend.services.inference import detection_uncertainty as u
    assert u([]) == 0.5 and u([0.1, 0.2]) == 0.5     # nothing (above the noise floor) found
    assert u([0.99]) < u([0.7]) < u([0.3])
    # a busy image with one doubtful object among many sure ones is NOT more uncertain than an
    # image where every detection is doubtful (the old max rule saturated on dense images)
    busy = [0.9] * 49 + [0.5]
    doubtful = [0.45, 0.4]
    assert u(busy) < u(doubtful)


def test_scoring_job_ranks_images(client, monkeypatch):
    from backend.routers import models as models_router
    scores = iter([0.2, 0.9])
    monkeypatch.setattr(models_router.inference, "resolve_weights", lambda n, db=None: n)
    monkeypatch.setattr(models_router.inference, "uncertainty", lambda *a, **k: next(scores))
    pid, _, ids = _project(client, n_images=2)
    r = client.post(f"/api/projects/{pid}/active-learning/score", json={"model_name": "yolo11n.pt"})
    assert r.status_code == 200
    job = client.get(f"/api/projects/{pid}/autolabel/{r.json()['id']}").json()
    assert job["status"] == "done" and job["images_labeled"] == 2
    unc = {i["id"]: i["uncertainty"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert sorted(unc.values()) == [0.2, 0.9]
    st = client.get(f"/api/projects/{pid}/active-learning").json()
    assert st["scored"] == 2 and st["model_name"] == "yolo11n.pt"
    # deleting an image removes its score
    client.delete(f"/api/projects/{pid}/images/{ids[0]}")
    assert client.get(f"/api/projects/{pid}/active-learning").json()["scored"] == 1


# ── SAM GPU memory handling ──────────────────────────────────────────
def test_find_all_retries_smaller_on_oom_then_reports_clearly(monkeypatch):
    import pytest
    from backend.services import sam
    sides = []

    def fake(image_path, model_name, box, text, conf, device, simplify, imgsz):
        sides.append(imgsz)
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(sam, "_predict_all", fake)
    with pytest.raises(sam.SamError, match="GPU out of memory"):
        sam.predict_all("x.jpg", "sam3.pt", box=[0, 0, 5, 5])
    assert sides == [1008, 644]        # SAM 3's native input size, then one retry smaller


def test_find_all_oom_retry_can_succeed(monkeypatch):
    from backend.services import sam
    calls = []

    def fake(image_path, model_name, box, text, conf, device, simplify, max_side):
        calls.append(max_side)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return [{"polygon": [[0, 0], [1, 0], [1, 1]], "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}, "score": 0.9, "label": ""}]

    monkeypatch.setattr(sam, "_predict_all", fake)
    assert len(sam.predict_all("x.jpg", "sam3.pt", box=[0, 0, 5, 5])) == 1 and len(calls) == 2


def test_warmup_endpoint_reports_sam_errors(client, monkeypatch):
    from backend.services import sam
    monkeypatch.setattr(sam, "warmup", lambda *a, **k: {"ok": True})
    assert client.post("/api/sam/warmup", json={}).json() == {"ok": True}

    def boom(*a, **k):
        raise sam.SamError("no weights")
    monkeypatch.setattr(sam, "warmup", boom)
    assert client.post("/api/sam/warmup", json={}).status_code == 422


def test_find_all_several_classes_in_one_job(client, monkeypatch):
    from backend.services import sam
    boxes = {"port": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "score": 0.9, "label": ""}],
             "cable": [{"polygon": [[50, 50], [80, 50], [80, 80], [50, 80]], "bbox": {"x": 50, "y": 50, "w": 30, "h": 30}, "score": 0.7, "label": ""},
                       # the same object the "port" phrase already found, but scored lower → dropped
                       {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "score": 0.5, "label": ""}]}
    seen = {}
    monkeypatch.setattr(sam, "predict_phrases", lambda path, model, phrases, **k: seen.update(p=phrases) or {p: boxes[p] for p in phrases})
    pid, c_port, (iid,) = _project(client)
    c_cable = client.post(f"/api/projects/{pid}/classes", json={"name": "cable"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/sam/find-all", json={"targets": [
        {"class_id": c_port, "text": "port"}, {"class_id": c_cable, "text": "cable"}]})
    assert r.status_code == 200
    job = client.get(f"/api/projects/{pid}/autolabel/{r.json()['id']}").json()
    assert job["status"] == "done" and job["annotations_added"] == 2
    assert seen["p"] == ["port", "cable"]
    anns = client.get(f"/api/projects/{pid}/images/{iid}/annotations").json()
    assert sorted(a["class_id"] for a in anns) == sorted([c_port, c_cable])


def test_find_all_needs_at_least_one_target(client):
    pid, cid, _ = _project(client)
    assert client.post(f"/api/projects/{pid}/sam/find-all", json={}).status_code == 400


# ── generic find-all → class-less proposals ──────────────────────────
def test_generic_find_all_stores_proposals_not_annotations(client, monkeypatch):
    from backend.services import sam
    seen = {}
    monkeypatch.setattr(sam, "predict_phrases",
                        lambda path, model, phrases, **k: seen.update(p=phrases) or
                        {phrases[0]: _MATCHES, phrases[-1]: [dict(_MATCHES[0], score=0.3)]})   # same object, lower score
    pid, cid, (i1, i2) = _project(client, n_images=2)
    client.put(f"/api/projects/{pid}/images/{i2}/annotations", json={"annotations": [
        {"class_id": cid, "kind": "bbox", "data": {"x": 0, "y": 0, "w": 10, "h": 10}}]})   # covers match #1 of image 2
    r = client.post(f"/api/projects/{pid}/sam/find-all", json={"mode": "generic", "phrases": ["component", "part"],
                                                                "only_unannotated": False})
    assert r.status_code == 200
    job = client.get(f"/api/projects/{pid}/autolabel/{r.json()['id']}").json()
    assert job["status"] == "done" and seen["p"] == ["component", "part"]

    ims = {i["id"]: i for i in client.get(f"/api/projects/{pid}/images").json()}
    assert ims[i1]["proposal_count"] == 2 and ims[i1]["annotation_count"] == 0 and ims[i1]["status"] == "unannotated"
    assert ims[i2]["proposal_count"] == 1                      # the one already annotated is skipped
    props = client.get(f"/api/projects/{pid}/images/{i1}/proposals").json()
    assert props[0]["score"] == 0.8 and len(props[0]["polygon"]) == 4 and props[0]["bbox"]["w"] == 10

    assert client.delete(f"/api/projects/{pid}/images/{i1}/proposals").status_code == 200
    assert client.get(f"/api/projects/{pid}/images/{i1}/proposals").json() == []


def test_generic_mode_needs_no_classes(client, monkeypatch):
    from backend.services import sam
    monkeypatch.setattr(sam, "predict_phrases", lambda path, model, phrases, **k: {p: [] for p in phrases})
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(50, 50), "image/png"))]
    client.post(f"/api/projects/{pid}/images", files=files)
    r = client.post(f"/api/projects/{pid}/sam/find-all", json={"mode": "generic"})
    assert r.status_code == 200      # default phrases; no class required
