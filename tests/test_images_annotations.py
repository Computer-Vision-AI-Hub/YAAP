"""Image upload/list/patch/delete + annotation replace."""
from conftest import make_png_bytes


def _project(client, task_type="detect"):
    return client.post("/api/projects", json={"name": "P", "task_type": task_type}).json()["id"]


def test_upload_creates_image_with_dimensions(client):
    pid = _project(client)
    files = [("files", ("a.png", make_png_bytes(64, 48), "image/png"))]
    r = client.post(f"/api/projects/{pid}/images", files=files)
    assert r.status_code == 200
    body = r.json()
    assert len(body["created"]) == 1
    assert body["created"][0]["width"] == 64
    assert body["created"][0]["height"] == 48
    assert body["created"][0]["status"] == "unannotated"
    assert body["skipped"] == []


def test_upload_skips_disallowed_extension(client):
    pid = _project(client)
    files = [("files", ("a.txt", b"not an image", "text/plain"))]
    r = client.post(f"/api/projects/{pid}/images", files=files).json()
    assert r["created"] == []
    assert r["skipped"] == ["a.txt"]


def test_upload_dedupes_filename_collisions(client):
    pid = _project(client)
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    n1 = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["filename"]
    n2 = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["filename"]
    assert n1 != n2


def test_list_images_filters_by_status(client):
    pid = _project(client)
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    assert len(client.get(f"/api/projects/{pid}/images").json()) == 1
    assert client.get(f"/api/projects/{pid}/images", params={"status": "approved"}).json() == []

    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "approved"})
    assert len(client.get(f"/api/projects/{pid}/images", params={"status": "approved"}).json()) == 1


def test_delete_image(client):
    pid = _project(client)
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]
    assert client.delete(f"/api/projects/{pid}/images/{iid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}/images/{iid}/file").status_code == 404


def test_replace_annotations_sets_status_annotated(client):
    pid = _project(client)
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(100, 100), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    payload = {"annotations": [{"class_id": cid, "kind": "bbox",
                                 "data": {"x": 1, "y": 2, "w": 10, "h": 10}}]}
    r = client.put(f"/api/projects/{pid}/images/{iid}/annotations", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "annotated"
    assert len(r.json()["annotations"]) == 1

    r = client.get(f"/api/projects/{pid}/images/{iid}/annotations")
    assert len(r.json()) == 1
    assert r.json()[0]["data"] == {"x": 1, "y": 2, "w": 10, "h": 10}


def test_replace_annotations_with_empty_list_reverts_to_unannotated(client):
    pid = _project(client)
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    payload = {"annotations": [{"class_id": cid, "kind": "bbox", "data": {"x": 0, "y": 0, "w": 5, "h": 5}}]}
    client.put(f"/api/projects/{pid}/images/{iid}/annotations", json=payload)

    r = client.put(f"/api/projects/{pid}/images/{iid}/annotations", json={"annotations": []})
    assert r.json()["status"] == "unannotated"


def test_replace_annotations_rejects_unknown_class(client):
    pid = _project(client)
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    payload = {"annotations": [{"class_id": 999, "kind": "bbox", "data": {}}]}
    r = client.put(f"/api/projects/{pid}/images/{iid}/annotations", json=payload)
    assert r.status_code == 400
