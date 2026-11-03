"""SSH training mode — settings, validation, job serialization and the
standalone remote script. No real ssh connection is made."""
import json

import yaml

from backend.db import SessionLocal, TrainJob
from backend.services import ssh_runner
from backend.services.remote_train import localize_data_yaml


def test_settings_roundtrip(client):
    assert client.get("/api/training/ssh/settings").json()["workdir"] == "~/yaap_jobs"
    body = {"host": "jetson.local", "user": "nx", "port": 2222, "python": "~/yaap-env/bin/python"}
    assert client.put("/api/training/ssh/settings", json=body).status_code == 200
    got = client.get("/api/training/ssh/settings").json()
    assert got["host"] == "jetson.local" and got["port"] == 2222


def test_settings_reject_option_injection(client):
    r = client.put("/api/training/ssh/settings", json={"host": "-oProxyCommand=evil"})
    assert r.status_code == 400


def test_test_endpoint_reports_missing_host(client):
    r = client.post("/api/training/ssh/test", json={}).json()
    assert r["ok"] is False and "host" in r["error"].lower()


def test_ssh_command_is_batch_mode():
    cmd = ssh_runner.ssh_cmd({**ssh_runner.DEFAULTS, "host": "h", "user": "u", "port": 2200,
                              "identity_file": "~/k"}, "true")
    assert "BatchMode=yes" in cmd and "-p" in cmd and cmd[-2] == "u@h"


def test_remove_remote_refuses_non_job_dir():
    import pytest
    with pytest.raises(ssh_runner.SshError):
        ssh_runner.remove_remote({**ssh_runner.DEFAULTS, "host": "h"}, "/home/u")


def test_job_serialization_shows_ssh_host(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    db = SessionLocal()
    params = {"mode": "ssh", "ssh": {**ssh_runner.DEFAULTS, "host": "jetson.local", "user": "nx"}}
    db.add(TrainJob(project_id=pid, version_id=1, model_arch="yolo11n.pt", status="done",
                    params=json.dumps(params)))
    db.commit()
    db.close()
    job = client.get(f"/api/projects/{pid}/train").json()[0]
    assert job["mode"] == "ssh" and job["ssh_host"] == "nx@jetson.local"


def test_remote_script_rewrites_data_yaml_path(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text(yaml.safe_dump({"path": "/somewhere/else", "train": "train/images"}))
    localize_data_yaml(str(y))
    assert yaml.safe_load(y.read_text())["path"] == str(tmp_path.resolve())
