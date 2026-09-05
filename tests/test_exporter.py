"""Dataset version generation — YOLO layout, and the additive COCO layout
fix (a "coco" version must stay trainable in-platform too, see
backend/services/exporter.py)."""
from pathlib import Path

from conftest import make_png_bytes


def _annotated_project(client, task_type="detect", n_images=3):
    pid = client.post("/api/projects", json={"name": "P", "task_type": task_type}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    image_ids = []
    for i in range(n_images):
        files = [("files", (f"img{i}.png", make_png_bytes(100, 100), "image/png"))]
        iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]
        kind = "polygon" if task_type == "segment" else "bbox"
        data = ({"points": [[10, 10], [50, 10], [50, 50], [10, 50]]} if kind == "polygon"
                else {"x": 10, "y": 10, "w": 30, "h": 20})
        client.put(f"/api/projects/{pid}/images/{iid}/annotations",
                   json={"annotations": [{"class_id": cid, "kind": kind, "data": data}]})
        client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
        image_ids.append(iid)
    return pid, cid, image_ids


def _create_version(client, pid, **cfg):
    r = client.post(f"/api/projects/{pid}/versions", json={"name": "v1", **cfg})
    assert r.status_code == 200
    vid = r.json()["id"]
    v = client.get(f"/api/projects/{pid}/versions").json()[0]
    assert v["id"] == vid
    return v


def _version_dir(pid, vid):
    from backend.config import DATA_DIR
    return Path(DATA_DIR) / "projects" / str(pid) / "versions" / f"v{vid}"


def test_yolo_export_builds_data_yaml_and_labels(client):
    pid, cid, _ = _annotated_project(client)
    v = _create_version(client, pid, format="yolo")
    assert v["status"] == "ready"
    assert v["stats"]["format"] == "yolo"

    vdir = _version_dir(pid, v["id"])
    assert (vdir / "data.yaml").exists()
    label_files = list((vdir / "train" / "labels").glob("*.txt")) + \
        list((vdir / "val" / "labels").glob("*.txt")) + \
        list((vdir / "test" / "labels").glob("*.txt"))
    assert label_files
    line = label_files[0].read_text().strip()
    parts = line.split()
    assert parts[0] == "0"                                    # first (only) class index
    assert len(parts) == 5                                    # class cx cy w h
    # bbox (10,10,30,20) on a 100x100 image -> cx=(10+15)/100, cy=(10+10)/100
    assert abs(float(parts[1]) - 0.25) < 1e-4
    assert abs(float(parts[2]) - 0.20) < 1e-4


def test_coco_export_is_additive_not_exclusive(client):
    """The bug we fixed: a 'coco' version must still contain a working
    data.yaml + images/labels (so it trains in-platform), plus the COCO
    json+flat-images layout on top — not one or the other."""
    pid, cid, _ = _annotated_project(client)
    v = _create_version(client, pid, format="coco")
    assert v["status"] == "ready", v["stats"]
    assert v["stats"]["format"] == "coco"

    vdir = _version_dir(pid, v["id"])
    # YOLO layout must exist regardless of format=coco
    assert (vdir / "data.yaml").exists()
    yolo_splits = [s for s in ("train", "val", "test") if (vdir / s / "images").exists()]
    assert yolo_splits
    for s in yolo_splits:
        assert list((vdir / s / "images").glob("*.jpg"))
        assert list((vdir / s / "labels").glob("*.txt"))

    # COCO layout must ALSO exist, flat images + json per split
    coco_splits = [s for s in ("train", "val", "test") if (vdir / s / "_annotations.coco.json").exists()]
    assert coco_splits
    for s in coco_splits:
        assert list((vdir / s).glob("*.jpg"))   # flat, not under images/


def test_coco_json_content_matches_annotations(client):
    pid, cid, _ = _annotated_project(client, n_images=1)
    # force everything into train, and keep the default 640 resize from
    # rescaling our 100x100 source image (which would scale the bbox too)
    v = _create_version(client, pid, format="coco", splits={"train": 1, "val": 0, "test": 0},
                        preprocess={"resize": 100, "mode": "stretch", "grayscale": False})
    vdir = _version_dir(pid, v["id"])

    import json
    coco = json.loads((vdir / "train" / "_annotations.coco.json").read_text())
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    assert coco["categories"] == [{"id": 1, "name": "cat", "supercategory": "none"}]
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["annotations"][0]["bbox"] == [10, 10, 30, 20]


def test_coco_format_rejected_for_classify_projects(client):
    pid, _, _ = _annotated_project(client, task_type="classify", n_images=0)
    r = client.post(f"/api/projects/{pid}/versions", json={"name": "v1", "format": "coco"})
    assert r.status_code == 200          # version row is created...
    v = client.get(f"/api/projects/{pid}/versions").json()[0]
    assert v["status"] == "failed"       # ...but the background build rejects it
    assert "classif" in v["stats"]["error"].lower()


def test_segment_task_writes_polygon_labels(client):
    pid, cid, _ = _annotated_project(client, task_type="segment", n_images=1)
    v = _create_version(client, pid, format="yolo")
    vdir = _version_dir(pid, v["id"])
    label_files = [f for s in ("train", "val", "test") for f in (vdir / s / "labels").glob("*.txt")]
    line = label_files[0].read_text().strip().split()
    assert line[0] == "0"
    assert len(line) == 1 + 4 * 2        # class + 4 (x,y) pairs for the square polygon


def test_no_approved_images_raises(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    client.post(f"/api/projects/{pid}/classes", json={"name": "cat"})
    r = client.post(f"/api/projects/{pid}/versions", json={"name": "v1"})
    vid = r.json()["id"]
    v = client.get(f"/api/projects/{pid}/versions").json()[0]
    assert v["id"] == vid
    assert v["status"] == "failed"
    assert "no approved images" in v["stats"]["error"].lower()
