"""Regression tests for the bugs found by the VisDrone end-to-end test (2026-11-08).
Each test names its bug report (report/bugs/<id>.md in the test run)."""
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from conftest import make_png_bytes

BOX = {"x": 10, "y": 10, "w": 20, "h": 20}


def _project(client, task="detect", classes=("car", "bus")):
    pid = client.post("/api/projects", json={"name": "P", "task_type": task}).json()["id"]
    cids = [client.post(f"/api/projects/{pid}/classes", json={"name": c}).json()["id"] for c in classes]
    return pid, cids


def _upload(client, pid, name="a.png", raw=None, mime="image/png"):
    r = client.post(f"/api/projects/{pid}/images", files=[("files", (name, raw or make_png_bytes(100, 80), mime))])
    return r.json()


def _put(client, pid, iid, anns, **extra):
    return client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": anns, **extra})


# ── sam-1: SAM 3 "find all" sees the image at its native 1008 px, not ultralytics' default 640 ──
def test_find_all_predictor_runs_at_sam3_native_size(monkeypatch):
    import ultralytics.models.sam as usam
    from backend.services import sam
    seen = {}

    class FakePredictor:
        def __init__(self, overrides):
            seen.update(overrides)

    monkeypatch.setattr(usam, "SAM3SemanticPredictor", FakePredictor)
    sam._load_semantic.cache_clear()
    try:
        sam._load_semantic("sam3.pt", "cpu")
    finally:
        sam._load_semantic.cache_clear()
    assert seen["imgsz"] == 1008


# ── sam-2: generic find-all without nouns looks for the project's classes ──
def test_generic_find_all_defaults_to_the_class_names(client, monkeypatch):
    from backend.routers import sam as sam_router
    pid, _ = _project(client, classes=("car", "bus"))
    _upload(client, pid)
    asked = []
    monkeypatch.setattr(sam_router.sam, "predict_phrases", lambda path, m, phrases, **k: asked.append(phrases) or {})
    r = client.post(f"/api/projects/{pid}/sam/find-all", json={"mode": "generic", "phrases": []})
    assert r.status_code == 200
    assert asked == [["car", "bus"]]


# ── bootstrap-6: one "database is locked" skips that image, the job goes on ──
def test_find_all_survives_a_locked_database_on_one_image(client, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from backend.routers import sam as sam_router
    pid, (car, _) = _project(client)
    _upload(client, pid, "a.png"); _upload(client, pid, "b.png")
    hit = {"bbox": BOX, "polygon": [[10, 10], [30, 10], [30, 30]], "score": 0.9, "label": ""}
    monkeypatch.setattr(sam_router.sam, "predict_phrases", lambda *a, **k: {"car": [hit]})
    real, calls = sam_router.jobs.claim_image, []

    def flaky(db, iid, seen):
        calls.append(iid)
        if len(calls) == 1:
            raise OperationalError("UPDATE images", {}, Exception("database is locked"))
        return real(db, iid, seen)

    monkeypatch.setattr(sam_router.jobs, "claim_image", flaky)
    jid = client.post(f"/api/projects/{pid}/sam/find-all",
                      json={"targets": [{"class_id": car, "text": "car"}]}).json()["id"]
    job = client.get(f"/api/projects/{pid}/autolabel/{jid}").json()
    assert job["status"] == "done" and job["processed"] == 2
    assert job["images_labeled"] == 1 and len(job["skipped"]) == 1
    assert "database busy" in job["log"]
    # what find-all writes is marked as SAM's (Bootstrap won't replace it by default)
    ims = {im["filename"]: im for im in client.get(f"/api/projects/{pid}/images").json()}
    anns = client.get(f"/api/projects/{pid}/images/{ims['b.png']['id']}/annotations").json()
    assert [a["origin"] for a in anns] == ["sam"]


# ── platform-1 / bootstrap-1: pretrained aliases use the preloaded file in data/models ──
def test_pretrained_alias_resolves_to_the_models_dir():
    from backend.config import MODELS_DIR
    from backend.services import inference
    assert inference.resolve_weights("yolo11n.pt") == str(MODELS_DIR / "yolo11n.pt")
    assert inference.alias_path("weight-file.pt") == "weight-file.pt"      # only known aliases are mapped


# ── bootstrap-5: a root-owned ultralytics config folder doesn't crash training ──
@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_training_worker_gets_a_writable_config_dir(tmp_path, monkeypatch):
    from backend.services import trainer
    cfg = tmp_path / "models"
    (cfg / "Ultralytics").mkdir(parents=True)
    (cfg / "Ultralytics" / "Arial.ttf").write_bytes(b"font")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(cfg))
    env, note = trainer.worker_env()
    assert env["YOLO_CONFIG_DIR"] == str(cfg) and note == ""               # writable: used as is
    os.chmod(cfg / "Ultralytics", 0o500)
    try:
        env, note = trainer.worker_env()
    finally:
        os.chmod(cfg / "Ultralytics", 0o700)
    alt = Path(env["YOLO_CONFIG_DIR"])
    assert alt != cfg and "not writable" in note
    assert (alt / "Ultralytics" / "Arial.ttf").read_bytes() == b"font"       # no font download needed


# ── bootstrap-3: ranking includes images that only hold unreviewed predictions ──
def test_ranking_scores_unreviewed_predictions_too(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "resolve_weights", lambda n, db=None: n)
    monkeypatch.setattr(models_router.inference, "uncertainty", lambda *a, **k: 0.5)
    pid, (car, _) = _project(client)
    ids = [_upload(client, pid, f"{n}.png")["created"][0]["id"] for n in "abc"]
    _put(client, pid, ids[1], [{"class_id": car, "kind": "bbox", "data": BOX, "source": "model"}])
    client.patch(f"/api/projects/{pid}/images/{ids[1]}", json={"status": "auto-annotated"})
    _put(client, pid, ids[2], [{"class_id": car, "kind": "bbox", "data": BOX}])           # checked by a person
    job = client.post(f"/api/projects/{pid}/active-learning/score", json={"model_name": "yolo11n.pt"}).json()
    assert job["total"] == 2


# ── bootstrap-7: a round keeps predictions made by SAM / Run a model unless asked ──
def test_bootstrap_does_not_replace_other_tools_predictions_by_default():
    from backend.db import Annotation, ImageAsset
    from backend.services import bootstrap
    im = ImageAsset(status="auto-annotated", filename="x.png")
    im.annotations = [Annotation(source="model", origin="sam", kind="bbox", class_id=1)]
    assert not bootstrap.is_target(im, True, True)
    assert bootstrap.is_target(im, True, True, other_tools=True)
    im.annotations = [Annotation(source="model", origin="bootstrap", kind="bbox", class_id=1)]
    assert bootstrap.is_target(im, True, True)
    im.annotations = [Annotation(source="model", origin="", kind="bbox", class_id=1)]      # older rows
    assert bootstrap.is_target(im, True, True)


def test_bootstrap_card_counts_other_tools_predictions(client):
    pid, (car, _) = _project(client)
    iid = _upload(client, pid)["created"][0]["id"]
    _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX, "source": "model", "origin": "sam"}])
    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "auto-annotated"})
    st = client.get(f"/api/projects/{pid}/bootstrap").json()
    assert st["unreviewed"] == 0 and st["other_unreviewed"] == 1


# ── platform-9 / platform-10: class names stay unique, export indices don't repeat ──
def test_class_rename_to_existing_name_is_refused(client):
    pid, (car, bus) = _project(client)
    r = client.patch(f"/api/projects/{pid}/classes/{bus}", json={"name": "car"})
    assert r.status_code == 409
    assert client.patch(f"/api/projects/{pid}/classes/{bus}", json={"name": "bus"}).status_code == 200


def test_new_class_after_a_delete_gets_a_fresh_index(client):
    pid, (a, b) = _project(client, classes=("a", "b"))
    client.post(f"/api/projects/{pid}/classes", json={"name": "c"})
    client.delete(f"/api/projects/{pid}/classes/{b}")
    client.post(f"/api/projects/{pid}/classes", json={"name": "d"})
    idx = {c["name"]: c["order_idx"] for c in client.get(f"/api/projects/{pid}").json()["classes"]}
    assert idx == {"a": 0, "c": 2, "d": 3}


# ── platform-3: broken geometry is refused at the door ──
@pytest.mark.parametrize("kind,data", [
    ("bbox", {}), ("bbox", {"x": 1, "y": 1, "w": -5, "h": 3}), ("bbox", {"x": 1, "y": 1, "w": 5, "h": 0}),
    ("polygon", {"points": [[1, 1], [2, 2]]}), ("polygon", {"points": "nope"}),
])
def test_annotation_put_refuses_broken_geometry(client, kind, data):
    pid, (car, _) = _project(client)
    iid = _upload(client, pid)["created"][0]["id"]
    assert _put(client, pid, iid, [{"class_id": car, "kind": kind, "data": data}]).status_code == 422


# ── annotator-6 / annotator-13: a stale editor can't silently delete what a job or another tab saved ──
def test_annotation_put_with_stale_ids_is_refused(client):
    pid, (car, _) = _project(client)
    iid = _upload(client, pid)["created"][0]["id"]
    first = _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}]).json()
    ids = [a["id"] for a in first["annotations"]]
    # someone else saves (another tab / a job): the ids change
    _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}] * 2)
    r = _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}], expect_ids=ids)
    assert r.status_code == 409
    assert len(client.get(f"/api/projects/{pid}/images/{iid}/annotations").json()) == 2   # nothing lost
    cur = [a["id"] for a in client.get(f"/api/projects/{pid}/images/{iid}/annotations").json()]
    assert _put(client, pid, iid, [], expect_ids=cur).status_code == 200


# ── annotator-7: un-approving an image without labels doesn't make it "labeled" ──
def test_unapproving_an_empty_image_returns_it_to_unannotated(client):
    pid, _ = _project(client)
    iid = _upload(client, pid)["created"][0]["id"]
    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
    r = client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "annotated"})
    assert r.json()["status"] == "unannotated"


# ── platform-2: one broken annotation no longer drops its image from a version silently ──
def test_version_keeps_an_image_with_one_broken_annotation(client):
    from backend.db import Annotation, SessionLocal
    pid, (car, _) = _project(client)
    iid = _upload(client, pid)["created"][0]["id"]
    _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}])
    db = SessionLocal()                     # a row from before the API validated geometry
    db.add(Annotation(image_id=iid, class_id=car, kind="bbox", data="{}")); db.commit(); db.close()
    client.post(f"/api/projects/{pid}/versions", json={"name": "v", "approved_only": False,
                                                       "splits": {"train": 1, "val": 0, "test": 0}})
    st = client.get(f"/api/projects/{pid}/versions").json()[0]["stats"]
    assert st["images"]["train"] == 1 and len(st["bad_annotations"]) == 1 and st["skipped"] == []
    # platform-14: the data.yaml inside the zip works on another machine (no absolute path)
    assert "path" not in __import__("yaml").safe_load(Path(st["path"], "data.yaml").read_text())
    # platform-16: the version remembers which image went where
    assert st["split_ids"]["train"] == [iid]


# ── platform-4 / platform-5: raw export ──
def test_raw_exports_dont_share_a_folder_and_same_stems_keep_their_labels(client):
    from backend.db import Project, SessionLocal
    from backend.services import exporter, storage
    pid, (car, _) = _project(client)
    a = _upload(client, pid, "same.png")["created"][0]["id"]
    b = _upload(client, pid, "same.jpg", raw=_jpeg(), mime="image/jpeg")["created"][0]["id"]
    _put(client, pid, a, [{"class_id": car, "kind": "bbox", "data": BOX}])
    _put(client, pid, b, [{"class_id": car, "kind": "bbox", "data": BOX}] * 2)
    db = SessionLocal()
    p = db.get(Project, pid)
    z1, z2 = exporter.export_raw(p), exporter.export_raw(p)      # e.g. a double-click on Download
    db.close()
    assert z1 != z2 and z1.exists() and z2.exists()
    names = zipfile.ZipFile(z1).namelist()
    labels = sorted(n for n in names if n.startswith("labels/"))
    assert len(labels) == 2
    lines = sorted(len(zipfile.ZipFile(z1).read(n).decode().split()) // 5 for n in labels)
    assert lines == [1, 2]
    exporter.remove_export(z1); exporter.remove_export(z2)
    assert not list(storage.project_dir(pid).glob("_raw_export*"))
    assert client.get(f"/api/projects/{pid}/export").status_code == 200


def _jpeg(w=60, h=40, exif=None):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, "JPEG", **({"exif": exif} if exif else {}))
    return buf.getvalue()


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, data in files.items():
            z.writestr(n, data)
    return buf.getvalue()


def _import(client, pid, raw):
    return client.post(f"/api/projects/{pid}/import", files={"file": ("d.zip", raw, "application/zip")})


# ── platform-6: the same file name in two splits keeps each split's own labels ──
def test_import_matches_labels_by_folder_not_just_name(client):
    pid, _ = _project(client, classes=())
    raw = _zip({"data.yaml": "names: [car, van]\n",
                "train/images/dup.png": make_png_bytes(), "train/labels/dup.txt": "0 0.5 0.5 0.2 0.2\n",
                "valid/images/dup.png": make_png_bytes(),
                "valid/labels/dup.txt": "1 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"})
    assert _import(client, pid, raw).status_code == 200
    names = {c["id"]: c["name"] for c in client.get(f"/api/projects/{pid}").json()["classes"]}
    got = {}
    for im in client.get(f"/api/projects/{pid}/images").json():
        anns = client.get(f"/api/projects/{pid}/images/{im['id']}/annotations").json()
        got[im["split"]] = sorted(names[a["class_id"]] for a in anns)
    assert got == {"train": ["car"], "val": ["van", "van"]}


# ── platform-7: a malformed label line is skipped and reported, the import goes on ──
def test_import_skips_malformed_label_lines(client):
    pid, _ = _project(client, classes=())
    raw = _zip({"data.yaml": "names: [car]\n",
                "images/m1.png": make_png_bytes(), "labels/m1.txt": "0 0.5 0.5 0.2 0.2\n",
                "images/m2.png": make_png_bytes(), "labels/m2.txt": "car 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.1 0.1\n",
                "images/m3.png": make_png_bytes(), "labels/m3.txt": "0 0.5 0.5 0.2 0.2\n"})
    r = _import(client, pid, raw).json()
    assert r["images_created"] == 3 and r["label_lines_skipped"] == 1
    assert "m2.txt:1" in r["label_line_errors"][0]


# ── platform-8: a corrupt upload leaves nothing behind ──
def test_corrupt_upload_leaves_no_file(client):
    from backend.services import storage
    pid, _ = _project(client)
    r = _upload(client, pid, "broken.png", raw=b"not an image")
    assert r["skipped"] == ["broken.png"]
    assert not storage.image_path(pid, "broken.png").exists()
    assert _upload(client, pid, "broken.png")["created"][0]["filename"] == "broken.png"


# ── platform-13: rotated phone photos are stored upright ──
def test_exif_rotated_upload_is_stored_upright(client):
    from backend.services import storage
    pid, _ = _project(client)
    exif = Image.Exif(); exif[0x0112] = 6                       # "rotate 90° CW to view"
    im = _upload(client, pid, "phone.jpg", raw=_jpeg(400, 200, exif.tobytes()), mime="image/jpeg")["created"][0]
    assert (im["width"], im["height"]) == (200, 400)
    with Image.open(storage.image_path(pid, im["filename"])) as f:
        assert f.size == (200, 400) and f.getexif().get(0x0112, 1) == 1


# ── platform-15: TIFF is stored as PNG, so the editor can show it ──
def test_tiff_upload_is_stored_as_png(client):
    from backend.services import storage
    pid, _ = _project(client)
    buf = io.BytesIO(); Image.new("I;16", (50, 30), 3000).save(buf, "TIFF")
    im = _upload(client, pid, "scan.tif", raw=buf.getvalue(), mime="image/tiff")["created"][0]
    assert im["filename"] == "scan.png" and (im["width"], im["height"]) == (50, 30)
    with Image.open(storage.image_path(pid, "scan.png")) as f:
        assert f.format == "PNG"


# ── platform-16: Test model uses the version's test split, even after the project changed ──
def test_test_model_uses_the_versions_own_test_split(client, monkeypatch):
    from backend.routers import models as models_router
    monkeypatch.setattr(models_router.inference, "predict", lambda *a, **k: [])
    monkeypatch.setattr(models_router.inference, "resolve_weights", lambda n, db=None: n)
    pid, (car, _) = _project(client)
    ids = []
    for i in range(10):
        iid = _upload(client, pid, f"i{i}.png")["created"][0]["id"]
        _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}])
        client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
        ids.append(iid)
    client.post(f"/api/projects/{pid}/versions", json={"name": "v", "splits": {"train": 0.5, "val": 0.2, "test": 0.3}})
    v = client.get(f"/api/projects/{pid}/versions").json()[0]
    test_ids = sorted(v["stats"]["split_ids"]["test"])
    client.patch(f"/api/projects/{pid}/images/{ids[0]}", json={"status": "annotated"})   # the project moves on
    r = client.post(f"/api/projects/{pid}/test-model", json={"version_id": v["id"], "model_name": "yolo11n.pt"})
    assert r.status_code == 200, r.text
    assert sorted(x["id"] for x in r.json()["images"]) == test_ids


# ── platform-11: closing one of two tabs doesn't shut YAAP down ──
def test_closing_one_of_two_tabs_does_not_arm_the_shutdown(client, monkeypatch):
    from backend.services import jobs
    monkeypatch.setattr(jobs, "_tabs", {})
    client.post("/api/system/here?tab=A"); client.post("/api/system/here?tab=B")
    r = client.post("/api/system/leaving?tab=A").json()
    assert r["ok"] is False and not jobs.LEAVE_MARKER.exists()
    assert client.post("/api/system/leaving?tab=B").json()["ok"] is True
    assert jobs.LEAVE_MARKER.exists()
    client.post("/api/system/here")
    assert not jobs.LEAVE_MARKER.exists()


# ── platform-12: a deferred shutdown doesn't survive the restart that killed its job ──
def test_restart_drops_a_shutdown_deferred_by_a_dead_job():
    from backend.services import jobs
    jobs.LEAVE_PENDING.write_text("1")
    jobs.fail_interrupted_jobs()
    assert not jobs.LEAVE_PENDING.exists()


# ── classification end-to-end test (2026-11-09) ──
def test_classify_images_hold_one_class_tag_and_no_boxes(client):
    pid, (car, bus) = _project(client, task="classify")
    iid = _upload(client, pid)["created"][0]["id"]
    tag = lambda c: {"class_id": c, "kind": "classification", "data": {}}  # noqa: E731
    assert _put(client, pid, iid, [tag(car), tag(bus)]).status_code == 422
    assert _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}]).status_code == 422
    assert _put(client, pid, iid, [tag(bus)]).status_code == 200
    det, (dcar, _) = _project(client)
    diid = _upload(client, det)["created"][0]["id"]
    assert _put(client, det, diid, [tag(dcar)]).status_code == 422          # no class tags on detect images


def test_bootstrap_uses_224_px_for_classification_by_default(client, monkeypatch):
    from backend.services import bootstrap
    monkeypatch.setattr(bootstrap.trainer, "launch", lambda db, job: job)
    pid, (car, bus) = _project(client, task="classify")
    for i in range(6):
        iid = _upload(client, pid, f"{i}.png")["created"][0]["id"]
        _put(client, pid, iid, [{"class_id": car if i % 2 else bus, "kind": "classification", "data": {}}])
    _upload(client, pid, "target.png")
    st = client.get(f"/api/projects/{pid}/bootstrap").json()
    assert st["default_imgsz"] == 224 and {c["name"] for c in st["few_examples"]} == {"car", "bus"}   # < 20 per class
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    from backend.db import SessionLocal, TrainJob
    db = SessionLocal()
    params = json.loads(db.get(TrainJob, run["train_job_id"]).params)
    db.close()
    assert params["imgsz"] == 224


# ── segmentation end-to-end test (2026-11-09): imported split pins can be cleared ──
def test_clear_splits_forgets_imported_pins(client):
    pid, _ = _project(client)
    ids = [_upload(client, pid, f"{n}.png")["created"][0]["id"] for n in "abc"]
    client.patch(f"/api/projects/{pid}/images/{ids[0]}", json={"split": "train"})
    client.patch(f"/api/projects/{pid}/images/{ids[1]}", json={"split": "val"})
    assert client.post(f"/api/projects/{pid}/images/clear-splits").json() == {"cleared": 2}
    assert {im["split"] for im in client.get(f"/api/projects/{pid}/images").json()} == {""}


# ── export round-trip test (2026-11-09) ──
def test_classify_raw_export_round_trip_keeps_the_class_order(client):
    pid, (car, bus, _van) = _project(client, task="classify", classes=("car", "bus", "van"))
    iid = _upload(client, pid)["created"][0]["id"]
    _put(client, pid, iid, [{"class_id": bus, "kind": "classification", "data": {}}])
    raw = client.get(f"/api/projects/{pid}/export").content
    assert "classes.txt" in zipfile.ZipFile(io.BytesIO(raw)).namelist()
    dst = client.post("/api/projects", json={"name": "Q", "task_type": "classify"}).json()["id"]
    assert client.post(f"/api/projects/{dst}/import", files={"file": ("r.zip", raw, "application/zip")}).status_code == 200
    assert [c["name"] for c in client.get(f"/api/projects/{dst}").json()["classes"]] == ["car", "bus", "van"]


def test_version_reports_labels_too_small_after_the_resize(client):
    pid, (car, _) = _project(client)
    iid = _upload(client, pid, raw=make_png_bytes(1000, 1000))["created"][0]["id"]
    _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": {"x": 10, "y": 10, "w": 3, "h": 3}},
                            {"class_id": car, "kind": "bbox", "data": {"x": 100, "y": 100, "w": 400, "h": 400}}])
    client.post(f"/api/projects/{pid}/versions", json={"name": "v", "approved_only": False, "splits": {"train": 1, "val": 0, "test": 0},
                                                       "preprocess": {"resize": 100, "mode": "stretch"}})
    st = client.get(f"/api/projects/{pid}/versions").json()[0]["stats"]
    assert st["annotations"] == 1 and st["tiny_dropped"] == 1


def test_coco_versions_use_roboflow_split_folders(client):
    import yaml
    pid, (car, _) = _project(client)
    for n in "abcdefghij":
        iid = _upload(client, pid, f"{n}.png")["created"][0]["id"]
        _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX}])
    for fmt in ("coco", "yolo"):
        client.post(f"/api/projects/{pid}/versions", json={"name": fmt, "format": fmt, "approved_only": False,
                                                           "splits": {"train": 0.6, "val": 0.2, "test": 0.2}})
    vs = {v["name"]: v for v in client.get(f"/api/projects/{pid}/versions").json()}
    coco, yolo = Path(vs["coco"]["stats"]["path"]), Path(vs["yolo"]["stats"]["path"])
    assert (coco / "valid" / "_annotations.coco.json").exists() and not (coco / "val").exists()   # RF-DETR reads valid/
    assert yaml.safe_load((coco / "data.yaml").read_text())["val"] == "valid/images"
    assert (yolo / "val" / "images").is_dir()                                                     # ultralytics' val/
    h = client.get(f"/api/projects/{pid}/versions/{vs['coco']['id']}/health").json()
    assert h["images"]["val"] == 2


# ── load test (2026-11-10): the image list must not cost one query per image ──
def test_listing_images_uses_a_fixed_number_of_queries(client):
    from sqlalchemy import event
    from backend.db import engine
    pid, (car, _) = _project(client)
    for i in range(30):
        iid = _upload(client, pid, f"{i}.png")["created"][0]["id"]
        _put(client, pid, iid, [{"class_id": car, "kind": "bbox", "data": BOX, "source": "model", "confidence": 0.5}])
    n = [0]
    count = lambda *a, **k: n.__setitem__(0, n[0] + 1)  # noqa: E731
    event.listen(engine, "before_cursor_execute", count)
    try:
        ims = client.get(f"/api/projects/{pid}/images").json()
    finally:
        event.remove(engine, "before_cursor_execute", count)
    assert len(ims) == 30 and all(i["annotation_count"] == 1 and i["min_conf"] == 0.5 for i in ims)
    assert n[0] <= 8, f"{n[0]} queries for 30 images"      # was 3 per image (5 000 images: 6 s)
