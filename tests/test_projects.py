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


def test_project_list_counts_and_constant_query_count(client):
    from sqlalchemy import event
    from backend.db import engine
    from conftest import make_png_bytes

    pid = client.post("/api/projects", json={"name": "P", "task_type": "detect"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "cat"}).json()["id"]
    ids = []
    for i in range(6):
        files = [("files", (f"i{i}.png", make_png_bytes(20, 20), "image/png"))]
        ids.append(client.post(f"/api/projects/{pid}/images", files=files).json()["created"][0]["id"])
    client.put(f"/api/projects/{pid}/images/{ids[0]}/annotations",
               json={"annotations": [{"class_id": cid, "kind": "bbox", "data": {"x": 1, "y": 1, "w": 5, "h": 5}}]})
    client.patch(f"/api/projects/{pid}/images/{ids[1]}", json={"status": "empty"})
    client.post(f"/api/projects/{pid}/versions", json={"name": "v1", "approved_only": False})

    def count_queries():
        n = [0]
        def on_exec(*a, **k): n[0] += 1
        event.listen(engine, "before_cursor_execute", on_exec)
        try:
            body = client.get("/api/projects").json()
        finally:
            event.remove(engine, "before_cursor_execute", on_exec)
        return n[0], body

    n_before, body = count_queries()
    p = next(x for x in body if x["id"] == pid)
    assert p["image_count"] == 6
    assert p["annotated_count"] == 2       # one annotated, one marked empty
    assert p["class_count"] == 1
    assert p["version_count"] == 1

    for i in range(10):                    # more unlabeled images must not add queries
        files = [("files", (f"x{i}.png", make_png_bytes(20, 20), "image/png"))]
        client.post(f"/api/projects/{pid}/images", files=files)
    n_after, _ = count_queries()
    assert n_after == n_before
