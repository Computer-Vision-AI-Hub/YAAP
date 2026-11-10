"""Background jobs next to a user who keeps working: auto-label / find-all never
overwrite an image edited meanwhile, "replace existing" never empties an image when
the model finds nothing, and closing the tab never kills a running job.
Models are mocked — the fakes below edit the image *during* inference to recreate
the race."""
from conftest import make_png_bytes

from backend.db import AutolabelJob, BootstrapRun, SessionLocal
from backend.services import jobs

BOX = {"x": 10, "y": 10, "w": 20, "h": 20}
PRED = {"class_name": "cat", "kind": "bbox", "data": {"x": 50, "y": 50, "w": 9, "h": 9}, "confidence": 0.9}


def _project(client, n=1):
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    files = [("files", (f"a{i}.png", make_png_bytes(100, 100), "image/png")) for i in range(n)]
    ids = [im["id"] for im in client.post(f"/api/projects/{pid}/images", files=files).json()["created"]]
    return pid, cid, ids


def _label(client, pid, iid, cid, data=BOX):
    client.put(f"/api/projects/{pid}/images/{iid}/annotations",
               json={"annotations": [{"class_id": cid, "kind": "bbox", "data": data}]})


def _anns(client, pid, iid):
    return client.get(f"/api/projects/{pid}/images/{iid}/annotations").json()


def _image(client, pid, iid):
    return next(im for im in client.get(f"/api/projects/{pid}/images").json() if im["id"] == iid)


# ── "replace existing" ───────────────────────────────────────────────
def test_replace_existing_keeps_an_approved_image_when_nothing_is_found(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [])
    pid, cid, (iid,) = _project(client)
    _label(client, pid, iid, cid)
    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt",
                                                              "replace_existing": True}).json()["id"]
    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert "existing annotations kept" in job["log"]
    assert _image(client, pid, iid)["status"] == "approved" and [a["data"] for a in _anns(client, pid, iid)] == [BOX]


def test_replace_existing_ignores_predictions_of_unknown_classes(client, monkeypatch):
    """A "dog" the project can't hold is no replacement either."""
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [dict(PRED, class_name="dog")])
    pid, cid, (iid,) = _project(client)
    _label(client, pid, iid, cid)
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt", "replace_existing": True})
    assert [a["data"] for a in _anns(client, pid, iid)] == [BOX]


def test_replace_existing_still_replaces_when_something_is_found(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [PRED])
    pid, cid, (iid,) = _project(client)
    _label(client, pid, iid, cid)
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt", "replace_existing": True})
    assert [a["data"] for a in _anns(client, pid, iid)] == [PRED["data"]]
    assert _image(client, pid, iid)["status"] == "auto-annotated"


# ── editing while a job runs ─────────────────────────────────────────
def test_autolabel_leaves_an_image_labeled_by_hand_meanwhile(client, monkeypatch):
    from backend.routers import models as models_router
    pid, cid, (iid,) = _project(client)

    def predict_while_user_labels(*a):
        _label(client, pid, iid, cid)         # the user saves while the model runs on this image
        return [PRED]
    monkeypatch.setattr(models_router.inference, "predict", predict_while_user_labels)
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"}).json()["id"]
    assert "changed by hand meanwhile" in client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["log"]
    assert _image(client, pid, iid)["status"] == "annotated" and [a["data"] for a in _anns(client, pid, iid)] == [BOX]


def test_autolabel_leaves_an_image_approved_meanwhile(client, monkeypatch):
    from backend.routers import models as models_router
    pid, cid, (iid,) = _project(client)
    _label(client, pid, iid, cid)

    def predict_while_user_approves(*a):
        client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
        return [PRED]
    monkeypatch.setattr(models_router.inference, "predict", predict_while_user_approves)
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt", "replace_existing": True})
    assert _image(client, pid, iid)["status"] == "approved" and [a["data"] for a in _anns(client, pid, iid)] == [BOX]


def test_find_all_leaves_an_image_labeled_by_hand_meanwhile(client, monkeypatch):
    from backend.services import sam
    pid, cid, (iid,) = _project(client)
    match = {"polygon": [[60, 60], [80, 60], [80, 80]], "bbox": {"x": 60, "y": 60, "w": 20, "h": 20},
             "score": 0.9, "label": ""}

    def phrases_while_user_labels(path, model, phrases, **k):
        _label(client, pid, iid, cid)
        return {p: [match] for p in phrases}
    monkeypatch.setattr(sam, "predict_phrases", phrases_while_user_labels)
    jid = client.post(f"/api/projects/{pid}/sam/find-all", json={"class_id": cid, "text": "cat"}).json()["id"]
    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert job["annotations_added"] == 0 and "changed by hand meanwhile" in job["log"]
    assert _image(client, pid, iid)["status"] == "annotated" and len(_anns(client, pid, iid)) == 1


# ── closing the tab ──────────────────────────────────────────────────
def _running_job(pid):
    db = SessionLocal()
    j = AutolabelJob(project_id=pid, model_name="yolo11n.pt", status="running")
    db.add(j)
    db.commit()
    jid = j.id
    db.close()
    return jid


def test_closing_the_tab_during_a_job_defers_the_shutdown_until_it_ends(client):
    pid, _, _ = _project(client)
    jid = _running_job(pid)
    r = client.post("/api/system/leaving").json()
    assert r == {"ok": False, "reason": "auto-label job running"}
    assert jobs.LEAVE_PENDING.exists() and not jobs.LEAVE_MARKER.exists()

    jobs.job_finished()                                      # still running → nothing yet
    assert not jobs.LEAVE_MARKER.exists()
    db = SessionLocal()
    db.get(AutolabelJob, jid).status = "done"
    db.commit()
    db.close()
    jobs.job_finished()
    assert jobs.LEAVE_MARKER.exists() and not jobs.LEAVE_PENDING.exists()


def test_a_real_job_arms_the_deferred_shutdown_when_it_finishes(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [])
    pid, _, _ = _project(client)
    jobs.LEAVE_PENDING.write_text("1")                      # the tab was closed while it ran
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"})
    assert jobs.LEAVE_MARKER.exists()


def test_coming_back_cancels_a_deferred_shutdown(client):
    pid, _, _ = _project(client)
    _running_job(pid)
    client.post("/api/system/leaving")
    client.post("/api/system/here")
    assert not jobs.LEAVE_PENDING.exists() and not jobs.LEAVE_MARKER.exists()


def test_jobs_cut_off_by_a_restart_are_marked_failed(client):
    pid, _, _ = _project(client)
    jid = _running_job(pid)
    owned = _running_job(pid)                               # a Bootstrap round's labeling job
    db = SessionLocal()
    db.add(BootstrapRun(project_id=pid, status="labeling", autolabel_job_id=owned))
    db.commit()
    db.close()
    jobs.fail_interrupted_jobs()
    assert client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["status"] == "failed"
    assert "Interrupted" in client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["error"]
    assert client.get(f"/api/projects/{pid}/autolabel/{owned}").json()["status"] == "running"


# ── one box per object ───────────────────────────────────────────────
A = {"x": 0, "y": 0, "w": 100, "h": 100}
NEAR = {"x": 5, "y": 5, "w": 100, "h": 100}       # IoU with A ≈ 0.82: the same object
HALF = {"x": 30, "y": 0, "w": 100, "h": 100}      # IoU with A ≈ 0.54
FAR = {"x": 300, "y": 0, "w": 50, "h": 50}


def test_drop_duplicates_keeps_one_box_per_object():
    new = [{"cls": "cat", "box": A, "conf": 0.9}, {"cls": "dog", "box": NEAR, "conf": 0.6},
           {"cls": "cat", "box": HALF, "conf": 0.5}, {"cls": "dog", "box": HALF, "conf": 0.45},
           {"cls": "cat", "box": FAR, "conf": 0.3}]
    kept, dropped = jobs.drop_duplicates(new, [], 0.7)
    # dog on NEAR: any class ≥ 0.7 → same object; cat on HALF: same class ≥ 0.5 → same object;
    # dog on HALF: another class at 0.54 — a different, overlapping object
    assert [(k["cls"], k["box"]) for k in kept] == [("cat", A), ("dog", HALF), ("cat", FAR)] and dropped == 2


def test_drop_duplicates_lets_existing_boxes_win_and_can_be_turned_off():
    existing = [{"cls": "dog", "box": A}]
    kept, dropped = jobs.drop_duplicates([{"cls": "cat", "box": NEAR, "conf": 0.99}], existing, 0.7)
    assert kept == [] and dropped == 1
    kept, dropped = jobs.drop_duplicates([{"cls": "cat", "box": NEAR, "conf": 0.99}], existing, 1)
    assert len(kept) == 1 and dropped == 0
    kept, _ = jobs.drop_duplicates([{"cls": "cat", "box": None, "conf": 0.4}, {"cls": "dog", "box": None, "conf": 0.8}],
                                   [], 0.7)                      # whole-image labels: one per image
    assert [k["cls"] for k in kept] == ["dog"]


def _two_class_project(client):
    pid, cat, (iid,) = _project(client)
    dog = client.post(f"/api/projects/{pid}/classes", json={"name": "dog"}).json()["id"]
    return pid, cat, dog, iid


def _box(cls_name, data, conf):
    return {"class_name": cls_name, "kind": "bbox", "data": data, "confidence": conf}


def test_autolabel_keeps_one_box_when_the_model_is_torn_between_classes(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict",
                        lambda *a: [_box("cat", A, 0.9), _box("dog", NEAR, 0.6), _box("cat", FAR, 0.8)])
    pid, cat, dog, iid = _two_class_project(client)
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"}).json()["id"]
    assert sorted((a["class_id"], a["data"]["x"]) for a in _anns(client, pid, iid)) == [(cat, 0), (cat, 300)]
    assert "1 duplicate(s) skipped" in client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["log"]


def test_autolabel_does_not_box_an_object_that_already_has_a_box(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [_box("cat", NEAR, 0.95), _box("cat", FAR, 0.8)])
    pid, cat, dog, iid = _two_class_project(client)
    _label(client, pid, iid, dog, A)                       # a person boxed it (as dog)
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"})
    assert sorted((a["class_id"], a["data"]["x"]) for a in _anns(client, pid, iid)) == [(cat, 300), (dog, 0)]


def test_autolabel_with_nothing_new_leaves_the_image_alone(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [_box("cat", NEAR, 0.95)])
    pid, cat, dog, iid = _two_class_project(client)
    _label(client, pid, iid, cat, A)
    jid = client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n.pt"}).json()["id"]
    assert _image(client, pid, iid)["status"] == "annotated" and len(_anns(client, pid, iid)) == 1
    assert "nothing new" in client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["log"]


def test_find_all_does_not_box_an_object_already_labeled_with_another_class(client, monkeypatch):
    from backend.services import sam
    match = {"polygon": [[5, 5], [105, 5], [105, 105]], "bbox": NEAR, "score": 0.9, "label": ""}
    monkeypatch.setattr(sam, "predict_phrases", lambda path, model, phrases, **k: {p: [match] for p in phrases})
    pid, cat, dog, iid = _two_class_project(client)
    _label(client, pid, iid, dog, A)
    jid = client.post(f"/api/projects/{pid}/sam/find-all", json={"class_id": cat, "text": "cat",
                                                                  "only_unannotated": False}).json()["id"]
    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert job["annotations_added"] == 0 and "1 duplicate(s) skipped" in job["log"]


# ── segmentation: outlines, not boxes ────────────────────────────────
UPPER = [[0, 0], [100, 0], [0, 100]]           # two triangles splitting one square:
LOWER = [[100, 0], [100, 100], [0, 100]]       # the same box, no shared area


def _poly(cls_name, pts, conf):
    return {"class_name": cls_name, "kind": "polygon", "data": {"points": pts}, "confidence": conf}


def test_outlines_are_compared_by_area_not_by_box():
    from backend.services.jobs import poly_iou, shape
    assert poly_iou(UPPER, LOWER) < 0.05 and poly_iou(UPPER, UPPER) > 0.95
    new = [shape("cat", "polygon", {"points": UPPER}, conf=0.9), shape("cat", "polygon", {"points": LOWER}, conf=0.8),
           shape("dog", "polygon", {"points": UPPER}, conf=0.5)]
    kept, dropped = jobs.drop_duplicates(new, [], 0.7)
    assert [k["poly"] for k in kept] == [UPPER, LOWER] and dropped == 1


def test_model_outlines_are_simplified_for_editing():
    import math
    from backend.services.inference import simplify_polygon
    circle = [[500 + 200 * math.cos(a / 180 * math.pi), 500 + 200 * math.sin(a / 180 * math.pi)] for a in range(360)]
    pts = simplify_polygon(circle)
    assert 8 <= len(pts) <= 40
    assert all(abs(math.dist(p, (500, 500)) - 200) < 3 for p in pts)     # still on the circle


def test_autolabel_in_a_segment_project_keeps_one_outline_per_object(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a: [
        _poly("cat", UPPER, 0.9), _poly("cat", LOWER, 0.8), _poly("dog", UPPER, 0.5)])
    pid = client.post("/api/projects", json={"name": "S", "task_type": "segment"}).json()["id"]
    for c in ("cat", "dog"):
        client.post(f"/api/projects/{pid}/classes", json={"name": c})
    files = [("files", ("a.png", make_png_bytes(200, 200), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]
    client.post(f"/api/projects/{pid}/autolabel", json={"model_name": "yolo11n-seg.pt"})
    anns = _anns(client, pid, iid)
    assert sorted(a["data"]["points"] for a in anns) == sorted([UPPER, LOWER])
    assert all(a["kind"] == "polygon" for a in anns)


def test_find_all_adds_a_new_class_to_annotated_images_and_keeps_the_rest(client, monkeypatch):
    """Imported dataset + a class added later: SAM 3 labels just that class on already-annotated
    images; the annotations that were there stay exactly as they were."""
    from backend.services import sam
    pid = client.post("/api/projects", json={"name": "S", "task_type": "segment"}).json()["id"]
    old = client.post(f"/api/projects/{pid}/classes", json={"name": "triangle"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(200, 200), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]
    client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": [
        {"class_id": old, "kind": "polygon", "data": {"points": UPPER}}]})
    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
    before = _anns(client, pid, iid)
    new = client.post(f"/api/projects/{pid}/classes", json={"name": "circle"}).json()["id"]
    circle = [[150, 150], [190, 150], [190, 190], [150, 190]]
    seen = {}
    monkeypatch.setattr(sam, "predict_phrases", lambda path, model, phrases, **k: seen.update(p=phrases) or {
        "circle": [{"polygon": circle, "bbox": {"x": 150, "y": 150, "w": 40, "h": 40}, "score": 0.9, "label": ""}]})
    jid = client.post(f"/api/projects/{pid}/sam/find-all", json={
        "targets": [{"class_id": new, "text": "circle"}], "only_unannotated": False}).json()["id"]
    assert client.get(f"/api/projects/{pid}/autolabel/{jid}").json()["annotations_added"] == 1
    assert seen["p"] == ["circle"]                          # only the new class is searched for
    after = _anns(client, pid, iid)
    assert [a for a in after if a["class_id"] == old] == before          # untouched: same id, data, source
    assert [a["data"]["points"] for a in after if a["class_id"] == new] == [circle]
    assert _image(client, pid, iid)["status"] == "auto-annotated"       # new predictions → back to Review
