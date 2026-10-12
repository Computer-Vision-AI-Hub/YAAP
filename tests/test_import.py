"""Dataset zip import — Roboflow-style YOLO exports."""
import io
import zipfile

import pytest

from conftest import make_png_bytes

SEG_LINE = "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n"   # polygon
BOX_LINE = "0 0.5 0.5 0.4 0.2\n"                  # cx cy w h


def _roboflow_zip(label_line):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.yaml", "nc: 1\nnames: ['dart']\n")
        for split in ("train", "valid"):
            zf.writestr(f"{split}/images/{split}1.png", make_png_bytes(100, 50))
            zf.writestr(f"{split}/labels/{split}1.txt", label_line)
    return buf.getvalue()


def _import(client, task, label_line):
    pid = client.post("/api/projects", json={"name": "P", "task_type": task}).json()["id"]
    r = client.post(f"/api/projects/{pid}/import",
                    files={"file": ("d.zip", _roboflow_zip(label_line), "application/zip")})
    assert r.status_code == 200
    ims = client.get(f"/api/projects/{pid}/images").json()
    anns = client.get(f"/api/projects/{pid}/images/{ims[0]['id']}/annotations").json()
    return ims, anns


def test_roboflow_valid_folder_maps_to_val_split(client):
    ims, _ = _import(client, "segment", SEG_LINE)
    assert sorted(i["split"] for i in ims) == ["train", "val"]
    assert all(i["status"] == "annotated" for i in ims)


def test_polygons_become_boxes_in_detect_project(client):
    _, anns = _import(client, "detect", SEG_LINE)
    assert len(anns) == 1 and anns[0]["kind"] == "bbox"
    assert anns[0]["data"] == pytest.approx({"x": 10, "y": 10, "w": 40, "h": 20})


def test_boxes_become_rectangles_in_segment_project(client):
    _, anns = _import(client, "segment", BOX_LINE)
    assert len(anns) == 1 and anns[0]["kind"] == "polygon"
    flat = [v for pt in anns[0]["data"]["points"] for v in pt]
    assert flat == pytest.approx([30, 20, 70, 20, 70, 30, 30, 30])
