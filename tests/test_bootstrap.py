"""Bootstrap: train on the human-labeled images, then label the rest.

No real training happens: trainer.launch() is swapped for a harmless shell
that only *looks* like the worker (same argv markers, so the pid checks accept
it), and the worker's labeling pass is called directly with inference.predict()
monkeypatched — everything around those two calls is real."""
import json
import os
import signal
import subprocess

import pytest
import yaml

from backend.config import RUNS_DIR
from backend.db import Annotation, BootstrapRun, ImageAsset, ModelWeight, SessionLocal, TrainJob
from backend.services import bootstrap, trainer
from conftest import make_png_bytes

BOX = {"x": 10, "y": 10, "w": 20, "h": 30}


class FakeLaunch:
    def __init__(self):
        self.script = "sleep 60; true"       # `; true` keeps sh from exec-ing sleep (argv must stay)
        self.procs = []

    def __call__(self, db, job):
        proc = subprocess.Popen(["sh", "-c", self.script, "backend.services.train_worker", "--job", str(job.id)],
                                start_new_session=True)
        self.procs.append(proc)
        job.pid, job.status = proc.pid, "running"
        db.commit()
        return job


@pytest.fixture
def launch(monkeypatch):
    fake = FakeLaunch()
    monkeypatch.setattr(trainer, "launch", fake)
    yield fake
    for p in fake.procs:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def _project(client, labeled=5, unlabeled=3, task="detect", classes=("cat",)):
    pid = client.post("/api/projects", json={"name": "P", "task_type": task}).json()["id"]
    cids = [client.post(f"/api/projects/{pid}/classes", json={"name": c}).json()["id"] for c in classes]
    files = [("files", (f"im{i}.png", make_png_bytes(100, 80), "image/png")) for i in range(labeled + unlabeled)]
    ids = [im["id"] for im in client.post(f"/api/projects/{pid}/images", files=files).json()["created"]]
    for n, iid in enumerate(ids[:labeled]):
        if task == "classify":
            ann = {"class_id": cids[n % len(cids)], "kind": "classification", "data": {}}
        elif task == "segment":
            ann = {"class_id": cids[0], "kind": "polygon", "data": {"points": [[10, 10], [40, 10], [25, 50]]}}
        else:
            ann = {"class_id": cids[0], "kind": "bbox", "data": BOX}
        client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": [ann]})
    return pid, cids, ids


def _set(iid, status=None, model_boxes=0, cid=None):
    db = SessionLocal()
    try:
        im = db.get(ImageAsset, iid)
        for _ in range(model_boxes):
            db.add(Annotation(image_id=iid, class_id=cid, kind="bbox", data=json.dumps(BOX),
                              source="model", confidence=0.3))
        if status:
            im.status = status
        db.commit()
    finally:
        db.close()


def test_auto_epochs_more_passes_for_fewer_images():
    assert bootstrap.auto_epochs(5) == 300
    assert bootstrap.auto_epochs(15) == 300       # VisDrone: 100 epochs F1 .38, 300 epochs .51
    assert bootstrap.auto_epochs(30) == 150
    assert bootstrap.auto_epochs(500) == 50


def test_auto_batch_shrinks_with_image_size():
    assert bootstrap.auto_batch(15, 640) == 8 and bootstrap.auto_batch(100, 640) == 16
    assert bootstrap.auto_batch(15, 1280) == 4 and bootstrap.auto_batch(100, 1280) == 8   # 16 @1280 ran out of memory
    assert bootstrap.auto_batch(15, 1920) == 2


def test_state_counts_what_trains_and_what_gets_labeled(client):
    pid, cids, ids = _project(client, labeled=5, unlabeled=5)
    _set(ids[5], status="empty")                                     # background
    _set(ids[6], status="auto-annotated", model_boxes=1, cid=cids[0])  # unreviewed guess
    _set(ids[7], status="needs_reannotation", model_boxes=1, cid=cids[0])
    s = client.get(f"/api/projects/{pid}/bootstrap").json()
    assert (s["labeled"], s["background"]) == (5, 1)
    assert (s["unannotated"], s["unreviewed"], s["rejected"]) == (2, 1, 1)
    assert s["default_model"] == "yolo11n.pt" and s["next_round"] == 1 and s["last"] is None


def test_start_needs_enough_labeled_images(client, launch):
    pid, _, _ = _project(client, labeled=3)
    r = client.post(f"/api/projects/{pid}/bootstrap", json={})
    assert r.status_code == 400 and "at least 5" in r.json()["detail"]
    assert not launch.procs


def test_start_needs_something_left_to_label(client, launch):
    pid, _, _ = _project(client, labeled=5, unlabeled=0)
    r = client.post(f"/api/projects/{pid}/bootstrap", json={})
    assert r.status_code == 400 and "nothing left" in r.json()["detail"]


def test_start_refuses_a_model_of_another_task(client, launch):
    pid, _, _ = _project(client)
    r = client.post(f"/api/projects/{pid}/bootstrap", json={"model_arch": "yolo11n-seg.pt"})
    assert r.status_code == 400


def test_start_builds_the_training_set_and_launches_the_worker(client, launch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=3)
    _set(ids[5], status="empty")
    r = client.post(f"/api/projects/{pid}/bootstrap", json={"conf": 0.3})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "training" and run["round"] == 1
    assert (run["train_images"], run["target_images"]) == (5, 2)

    db = SessionLocal()
    job = db.get(TrainJob, run["train_job_id"])
    params = json.loads(job.params)
    db.close()
    assert params["bootstrap_run"] == run["id"] and params["epochs"] == 300
    assert params["extra"] == {"val": False} and params["weight_name"].startswith("Bootstrap r1 · yolo11n")
    data = yaml.safe_load(open(params["data"]))
    assert data["names"] == {0: "cat"} and data["train"] == data["val"] == "images"
    ds = RUNS_DIR / f"job_{job.id}" / "dataset"
    assert len(list((ds / "images").iterdir())) == 6                  # 5 labeled + 1 background
    labels = {p.name: p.read_text() for p in (ds / "labels").iterdir()}
    assert labels[f"{ids[5]}_im5.txt"] == ""
    cx, cy = (BOX["x"] + BOX["w"] / 2) / 100, (BOX["y"] + BOX["h"] / 2) / 80
    assert labels[f"{ids[0]}_im0.txt"] == f"0 {cx:.6f} {cy:.6f} {20 / 100:.6f} {30 / 80:.6f}"

    jobs = client.get(f"/api/projects/{pid}/train").json()
    assert jobs[0]["bootstrap_round"] == 1
    assert client.get(f"/api/projects/{pid}/bootstrap").json()["last"]["status"] == "training"
    assert client.post(f"/api/projects/{pid}/bootstrap", json={}).status_code == 409   # one round at a time

    db = SessionLocal()                     # labeling phase: the TrainJob is done, the round is not
    db.get(TrainJob, job.id).status = "done"
    db.get(BootstrapRun, run["id"]).status = "labeling"
    db.commit()
    db.close()
    assert client.post("/api/system/leaving").json() == {"ok": False, "reason": "bootstrap running"}


def test_labeling_pass_fills_targets_and_replaces_unreviewed_guesses(client, launch, monkeypatch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=4)
    _set(ids[7], status="auto-annotated", model_boxes=2, cid=cids[0])   # last round's guess
    _set(ids[8], status="auto-annotated", model_boxes=1, cid=cids[0])   # new model finds nothing here
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()

    def fake_predict(model_name, path, task, conf, iou, device, db):
        assert model_name == f"weight:{wid}"
        if path.endswith("im5.png"):          # meanwhile, the user labels im6 by hand
            client.put(f"/api/projects/{pid}/images/{ids[6]}/annotations",
                       json={"annotations": [{"class_id": cids[0], "kind": "bbox", "data": BOX}]})
        if path.endswith("im8.png"):
            return []
        return [{"class_name": "cat", "kind": "bbox", "data": BOX, "confidence": 0.8},
                {"class_name": "dog", "kind": "bbox", "data": BOX, "confidence": 0.9}]   # no such class: dropped
    monkeypatch.setattr(bootstrap.inference, "predict", fake_predict)

    db = SessionLocal()
    w = ModelWeight(project_id=pid, name="Bootstrap r1", task="detect", path="/nonexistent.pt", source="trained")
    db.add(w)
    db.commit()
    wid = w.id
    bootstrap.label_phase(db, run["id"], wid)
    db.close()

    done = client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()
    assert done["status"] == "done" and done["weight_name"] == "Bootstrap r1"
    assert sorted(done["labeled_image_ids"]) == [ids[5], ids[7]]
    assert done["autolabel"]["status"] == "done" and "by hand meanwhile" in done["autolabel"]["log"]
    assert done["to_review"] == 2

    ims = {im["id"]: im for im in client.get(f"/api/projects/{pid}/images").json()}
    assert ims[ids[5]]["status"] == "auto-annotated" and ims[ids[5]]["annotation_count"] == 1
    assert ims[ids[6]]["status"] == "annotated"                        # the hand label survived
    assert ims[ids[7]]["annotation_count"] == 1                        # 2 old guesses → 1 new one
    assert ims[ids[8]]["status"] == "auto-annotated" and ims[ids[8]]["annotation_count"] == 1   # kept
    assert ims[ids[0]]["annotation_count"] == 1 and ims[ids[0]]["status"] == "annotated"

    s = client.get(f"/api/projects/{pid}/bootstrap").json()
    assert s["next_round"] == 2 and s["new_since_last"] == 1          # im6 was labeled during the round


def test_unreviewed_guesses_are_kept_when_refresh_is_off(client, launch, monkeypatch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=2)
    _set(ids[6], status="auto-annotated", model_boxes=2, cid=cids[0])
    run = client.post(f"/api/projects/{pid}/bootstrap", json={"refresh_unreviewed": False}).json()
    assert run["target_images"] == 1
    monkeypatch.setattr(bootstrap.inference, "predict", lambda *a: [])
    db = SessionLocal()
    bootstrap.label_phase(db, run["id"], 1)
    db.close()
    ims = {im["id"]: im for im in client.get(f"/api/projects/{pid}/images").json()}
    assert ims[ids[6]]["annotation_count"] == 2


def test_stop_kills_the_worker_and_marks_everything_stopped(client, launch):
    pid, _, _ = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    r = client.post(f"/api/projects/{pid}/bootstrap/{run['id']}/stop").json()
    assert r["status"] == "stopped"
    assert launch.procs[0].wait(timeout=5) == -signal.SIGTERM
    assert client.get(f"/api/projects/{pid}/train").json()[0]["status"] == "stopped"
    db = SessionLocal()
    bootstrap.fail(db, run["id"], "late worker error")                 # must not overwrite "stopped"
    assert db.get(BootstrapRun, run["id"]).status == "stopped"
    db.close()


def test_a_worker_that_died_silently_fails_its_run(client, launch):
    launch.script = "true"
    pid, _, _ = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    launch.procs[0].wait()
    r = client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()
    assert r["status"] == "failed" and "unexpectedly" in r["error"]
    assert client.post(f"/api/projects/{pid}/bootstrap", json={}).status_code == 200   # free again


def test_segment_set_keeps_polygons(client, launch):
    pid, _, ids = _project(client, task="segment")
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    ds = RUNS_DIR / f"job_{run['train_job_id']}" / "dataset"
    line = (ds / "labels" / f"{ids[0]}_im0.txt").read_text().split()
    assert line[0] == "0" and len(line) == 1 + 2 * 3


def test_classify_set_is_one_folder_per_class_with_val(client, launch):
    pid, _, _ = _project(client, task="classify", classes=("cat", "a/b"))
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    assert run["model_arch"] == "yolo11n-cls.pt"
    ds = RUNS_DIR / f"job_{run['train_job_id']}" / "dataset"
    for split in ("train", "val"):
        assert sorted(p.name for p in (ds / split).iterdir()) == ["a_b", "cat"]
        assert sum(len(list(d.iterdir())) for d in (ds / split).iterdir()) == 5


def test_classify_needs_two_classes(client, launch):
    pid, _, _ = _project(client, task="classify", classes=("cat",))
    r = client.post(f"/api/projects/{pid}/bootstrap", json={})
    assert r.status_code == 400 and "two classes" in r.json()["detail"]


def _weight(pid):
    db = SessionLocal()
    w = ModelWeight(project_id=pid, name="Bootstrap r1", task="detect", path="/nonexistent.pt", source="trained")
    db.add(w)
    db.commit()
    wid = w.id
    db.close()
    return wid


def _label(run_id, wid):
    db = SessionLocal()
    try:
        bootstrap.label_phase(db, run_id, wid)
    finally:
        db.close()


def test_training_set_is_a_copy_and_is_removed_after_training(client, launch, monkeypatch):
    """ultralytics re-saves JPEGs it thinks are corrupt in place — never on the originals."""
    from backend.services import storage
    pid, _, ids = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    ds = RUNS_DIR / f"job_{run['train_job_id']}" / "dataset"
    src = storage.image_path(pid, "im0.png")
    assert os.stat(ds / "images" / f"{ids[0]}_im0.png").st_ino != os.stat(src).st_ino
    monkeypatch.setattr(bootstrap.inference, "predict", lambda *a: [])
    _label(run["id"], _weight(pid))
    assert not ds.exists()


def test_a_hand_label_saved_while_its_image_is_predicted_wins(client, launch, monkeypatch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=1)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()

    def predict_while_user_labels(*a):      # the user's SAM label (saved as source "model") lands mid-inference
        client.put(f"/api/projects/{pid}/images/{ids[5]}/annotations", json={"annotations": [
            {"class_id": cids[0], "kind": "bbox", "data": BOX, "source": "model", "confidence": 0.97}]})
        return [{"class_name": "cat", "kind": "bbox", "data": {"x": 1, "y": 1, "w": 5, "h": 5}, "confidence": 0.8}]
    monkeypatch.setattr(bootstrap.inference, "predict", predict_while_user_labels)
    _label(run["id"], _weight(pid))

    im = next(i for i in client.get(f"/api/projects/{pid}/images").json() if i["id"] == ids[5])
    anns = client.get(f"/api/projects/{pid}/images/{ids[5]}/annotations").json()
    assert im["status"] == "annotated" and [a["data"] for a in anns] == [BOX]


def test_accept_refuses_predictions_the_reviewer_did_not_see(client):
    pid, cids, ids = _project(client, labeled=0, unlabeled=1)
    _set(ids[0], status="auto-annotated", model_boxes=1, cid=cids[0])
    shown = [a["id"] for a in client.get(f"/api/projects/{pid}/images/{ids[0]}/annotations").json()]
    _set(ids[0], model_boxes=1, cid=cids[0])                 # a new round added a box meanwhile
    r = client.patch(f"/api/projects/{pid}/images/{ids[0]}", json={"status": "approved", "expect_annotations": shown})
    assert r.status_code == 409
    current = [a["id"] for a in client.get(f"/api/projects/{pid}/images/{ids[0]}/annotations").json()]
    r = client.patch(f"/api/projects/{pid}/images/{ids[0]}", json={"status": "approved", "expect_annotations": current})
    assert r.status_code == 200 and r.json()["status"] == "approved"


def test_a_round_being_launched_is_not_mistaken_for_a_dead_one(client, monkeypatch):
    monkeypatch.setattr(trainer, "launch", lambda db, job: job)      # worker not started (yet)
    pid, _, _ = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    assert client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()["status"] == "training"
    assert client.post(f"/api/projects/{pid}/bootstrap", json={}).status_code == 409


def test_a_recycled_pid_is_not_taken_for_the_worker(client, launch):
    """After a container restart the stored pid can belong to any process — even the server."""
    pid, _, _ = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    db = SessionLocal()
    db.get(TrainJob, run["train_job_id"]).pid = os.getpid()             # alive, but not a worker
    db.commit()
    db.close()
    r = client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()
    assert r["status"] == "failed" and "unexpectedly" in r["error"]


def test_stopping_or_deleting_the_train_job_stops_the_round(client, launch):
    pid, _, _ = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    client.post(f"/api/projects/{pid}/train/{run['train_job_id']}/stop")
    assert launch.procs[0].wait(timeout=5) == -signal.SIGTERM
    assert client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()["status"] == "stopped"

    run2 = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    db = SessionLocal()                     # labeling phase: the TrainJob row already says done
    db.get(TrainJob, run2["train_job_id"]).status = "done"
    db.get(BootstrapRun, run2["id"]).status = "labeling"
    db.commit()
    db.close()
    client.delete(f"/api/projects/{pid}/train/{run2['train_job_id']}")
    assert launch.procs[1].wait(timeout=5) == -signal.SIGTERM
    assert client.get(f"/api/projects/{pid}/bootstrap/{run2['id']}").json()["status"] == "stopped"


def test_leaving_is_not_blocked_by_a_round_whose_worker_died(client, launch):
    launch.script = "true"
    pid, _, _ = _project(client)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    launch.procs[0].wait()
    db = SessionLocal()
    db.get(TrainJob, run["train_job_id"]).status = "done"
    db.get(BootstrapRun, run["id"]).status = "labeling"
    db.commit()
    db.close()
    assert client.post("/api/system/leaving").json()["ok"] is True
    client.post("/api/system/here")


def test_one_unreadable_image_does_not_end_the_round(client, launch, monkeypatch):
    pid, _, ids = _project(client, labeled=5, unlabeled=2)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()

    def predict(model, path, *a):
        if path.endswith("im5.png"):
            raise ValueError("cannot decode")
        return [{"class_name": "cat", "kind": "bbox", "data": BOX, "confidence": 0.9}]
    monkeypatch.setattr(bootstrap.inference, "predict", predict)
    _label(run["id"], _weight(pid))
    r = client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()
    assert r["status"] == "done" and r["labeled_image_ids"] == [ids[6]]
    assert r["autolabel"]["skipped"] == ["im5.png"]


def test_a_round_where_every_image_fails_is_reported_failed(client, launch, monkeypatch):
    pid, _, _ = _project(client, labeled=5, unlabeled=2)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    monkeypatch.setattr(bootstrap.inference, "predict", lambda *a: (_ for _ in ()).throw(RuntimeError("CUDA error")))
    db = SessionLocal()
    with pytest.raises(bootstrap.BootstrapError, match="CUDA error"):
        bootstrap.label_phase(db, run["id"], _weight(pid))
    bootstrap.fail(db, run["id"], "Labeling failed")
    db.close()
    assert client.get(f"/api/projects/{pid}/bootstrap/{run['id']}").json()["status"] == "failed"


def test_classes_renamed_mid_round_still_map(client, launch, monkeypatch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=1)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    client.patch(f"/api/projects/{pid}/classes/{cids[0]}", json={"name": "kitten"})
    monkeypatch.setattr(bootstrap.inference, "predict",
                        lambda *a: [{"class_name": "cat", "kind": "bbox", "data": BOX, "confidence": 0.9}])
    _label(run["id"], _weight(pid))
    anns = client.get(f"/api/projects/{pid}/images/{ids[5]}/annotations").json()
    assert [a["class_id"] for a in anns] == [cids[0]]


def test_classify_rounds_respect_min_confidence(client, launch, monkeypatch):
    pid, cids, ids = _project(client, task="classify", classes=("cat", "dog"), labeled=5, unlabeled=2)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={"conf": 0.5}).json()
    confs = iter([0.3, 0.9])
    monkeypatch.setattr(bootstrap.inference, "predict", lambda *a: [
        {"class_name": "dog", "kind": "classification", "data": {}, "confidence": next(confs)}])
    _label(run["id"], _weight(pid))
    ims = {im["id"]: im for im in client.get(f"/api/projects/{pid}/images").json()}
    assert ims[ids[5]]["status"] == "unannotated" and ims[ids[6]]["status"] == "auto-annotated"


def test_the_next_round_relabels_rejected_images(client, launch, monkeypatch):
    """Rejected predictions get the newer model's guess (and go back to Review) — boxes a
    person drew on a rejected image are never replaced."""
    pid, cids, ids = _project(client, labeled=5, unlabeled=3)
    _set(ids[5], status="needs_reannotation", model_boxes=2, cid=cids[0])
    db = SessionLocal()
    db.add(Annotation(image_id=ids[6], class_id=cids[0], kind="bbox", data=json.dumps(BOX), source="manual"))
    db.get(ImageAsset, ids[6]).status = "needs_reannotation"
    db.commit()
    db.close()
    assert client.get(f"/api/projects/{pid}/bootstrap").json()["rejected"] == 1
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    assert run["target_images"] == 2                       # ids[7] (unlabeled) + ids[5] (rejected guess)
    monkeypatch.setattr(bootstrap.inference, "predict",
                        lambda *a: [{"class_name": "cat", "kind": "bbox", "data": BOX, "confidence": 0.9}])
    _label(run["id"], _weight(pid))
    ims = {im["id"]: im for im in client.get(f"/api/projects/{pid}/images").json()}
    assert ims[ids[5]]["status"] == "auto-annotated" and ims[ids[5]]["annotation_count"] == 1
    assert ims[ids[6]]["status"] == "needs_reannotation" and ims[ids[6]]["annotation_count"] == 1


def test_rejected_images_wait_for_a_person_when_asked(client, launch, monkeypatch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=2)
    _set(ids[5], status="needs_reannotation", model_boxes=2, cid=cids[0])
    run = client.post(f"/api/projects/{pid}/bootstrap", json={"relabel_rejected": False}).json()
    assert run["target_images"] == 1
    monkeypatch.setattr(bootstrap.inference, "predict",
                        lambda *a: [{"class_name": "cat", "kind": "bbox", "data": BOX, "confidence": 0.9}])
    _label(run["id"], _weight(pid))
    im = next(i for i in client.get(f"/api/projects/{pid}/images").json() if i["id"] == ids[5])
    assert im["status"] == "needs_reannotation" and im["annotation_count"] == 2


def test_a_round_keeps_one_box_per_object(client, launch, monkeypatch):
    pid, cids, ids = _project(client, labeled=5, unlabeled=1)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    shifted = dict(BOX, x=BOX["x"] + 2)                    # the same object, found twice
    monkeypatch.setattr(bootstrap.inference, "predict", lambda *a: [
        {"class_name": "cat", "kind": "bbox", "data": BOX, "confidence": 0.9},
        {"class_name": "cat", "kind": "bbox", "data": shifted, "confidence": 0.5}])
    _label(run["id"], _weight(pid))
    anns = client.get(f"/api/projects/{pid}/images/{ids[5]}/annotations").json()
    assert [a["data"] for a in anns] == [BOX]


def test_a_segment_round_saves_outlines(client, launch, monkeypatch):
    pid, cids, ids = _project(client, task="segment", labeled=5, unlabeled=1)
    run = client.post(f"/api/projects/{pid}/bootstrap", json={}).json()
    assert run["model_arch"] == "yolo11n-seg.pt"
    upper, lower = [[0, 0], [60, 0], [0, 60]], [[60, 0], [60, 60], [0, 60]]   # one box, two objects
    monkeypatch.setattr(bootstrap.inference, "predict", lambda *a: [
        {"class_name": "cat", "kind": "polygon", "data": {"points": upper}, "confidence": 0.9},
        {"class_name": "cat", "kind": "polygon", "data": {"points": lower}, "confidence": 0.8}])
    _label(run["id"], _weight(pid))
    anns = client.get(f"/api/projects/{pid}/images/{ids[5]}/annotations").json()
    assert sorted(a["data"]["points"] for a in anns) == sorted([upper, lower])


def test_state_warns_about_classes_with_few_examples(client):
    pid, cids, ids = _project(client, labeled=5, unlabeled=1, classes=("cat", "dog"))
    s = client.get(f"/api/projects/{pid}/bootstrap").json()
    assert s["few_examples"] == [{"name": "dog", "count": 0}]       # all 5 hand labels are cats
