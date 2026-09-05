"""Project + class CRUD."""
from conftest import make_png_bytes


def test_empty_status_counts_toward_annotated_count(client):
    """A "Mark empty" image has zero real annotation rows but should still
    count as resolved/labeled, not pending — see routers/projects.py."""
    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    files = [("files", ("a.png", make_png_bytes(), "image/png"))]
    iid = client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"]

    assert client.get(f"/api/projects/{pid}").json()["annotated_count"] == 0
    client.patch(f"/api/projects/{pid}/images/{iid}", json={"status": "empty"})
    assert client.get(f"/api/projects/{pid}").json()["annotated_count"] == 1


def test_create_and_list_project(client):
    r = client.post("/api/projects", json={"name": "Cats", "task_type": "detect"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Cats"
    assert body["task_type"] == "detect"
    assert body["image_count"] == 0
    assert body["class_count"] == 0

    r = client.get("/api/projects")
    assert [p["name"] for p in r.json()] == ["Cats"]


def test_create_project_defaults_untitled_name(client):
    r = client.post("/api/projects", json={"name": "   "})
    assert r.status_code == 200
    assert r.json()["name"] == "Untitled"


def test_get_missing_project_404(client):
    r = client.get("/api/projects/999")
    assert r.status_code == 404


def test_delete_project(client):
    pid = client.post("/api/projects", json={"name": "Temp"}).json()["id"]
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_add_class_and_reject_duplicate(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]

    r = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"})
    assert r.status_code == 200
    assert r.json()["name"] == "cat"

    r = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"})
    assert r.status_code == 409


def test_add_class_rejects_empty_name(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/classes", json={"name": "   "})
    assert r.status_code == 400


def test_class_gets_next_palette_color_when_unset(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    c1 = client.post(f"/api/projects/{pid}/classes", json={"name": "a"}).json()
    c2 = client.post(f"/api/projects/{pid}/classes", json={"name": "b"}).json()
    assert c1["color"] != c2["color"]
    assert c1["order_idx"] == 0 and c2["order_idx"] == 1


def test_update_and_delete_class(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "a"}).json()["id"]

    r = client.patch(f"/api/projects/{pid}/classes/{cid}", json={"name": "b"})
    assert r.json()["name"] == "b"

    assert client.delete(f"/api/projects/{pid}/classes/{cid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}").json()["class_count"] == 0
