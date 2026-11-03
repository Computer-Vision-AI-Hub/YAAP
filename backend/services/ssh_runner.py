"""SSH training mode: connection settings, ssh/rsync helpers, and the remote
job flow driven by train_worker.py (the worker stays local and keeps all DB work).

Key-based auth only — every ssh call runs with BatchMode=yes, so a missing
key fails fast instead of hanging on a password prompt.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import DATA_DIR

SETTINGS_PATH = DATA_DIR / "ssh.json"
KNOWN_HOSTS = DATA_DIR / "ssh_known_hosts"
REMOTE_SCRIPT = Path(__file__).with_name("remote_train.py")
DEFAULTS = {"host": "", "user": "", "port": 22, "identity_file": "",
            "python": "~/yaap-env/bin/python", "workdir": "~/yaap_jobs", "cleanup": False}

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")


class SshError(RuntimeError):
    pass


# ── settings ─────────────────────────────────────────────────────────
def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        data = {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save_settings(s: dict) -> dict:
    clean = {**DEFAULTS, **{k: s[k] for k in DEFAULTS if k in s}}
    clean["host"] = (clean["host"] or "").strip()
    clean["user"] = (clean["user"] or "").strip()
    clean["identity_file"] = (clean["identity_file"] or "").strip()
    clean["port"] = int(clean["port"] or 22)
    clean["cleanup"] = bool(clean["cleanup"])
    validate(clean, need_host=False)
    SETTINGS_PATH.write_text(json.dumps(clean, indent=2))
    return clean


def validate(s: dict, need_host: bool = True):
    if need_host and not s["host"]:
        raise SshError("SSH host is not configured.")
    if s["host"] and not _HOST_RE.match(s["host"]):
        raise SshError("Invalid SSH host.")
    if s["user"] and not _USER_RE.match(s["user"]):
        raise SshError("Invalid SSH user.")
    if not 0 < int(s["port"]) < 65536:
        raise SshError("Invalid SSH port.")


def host_label(s: dict) -> str:
    return f"{s['user']}@{s['host']}" if s["user"] else s["host"]


# ── ssh / rsync primitives ───────────────────────────────────────────
def _opts(s: dict, port_flag: str = "-p") -> list[str]:
    o = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",      # ~/.ssh may be a read-only mount
         "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
    if int(s["port"]) != 22:             # leave 22 unset so ~/.ssh/config can supply its own Port
        o += [port_flag, str(s["port"])]
    if s["identity_file"]:
        o += ["-i", os.path.expanduser(s["identity_file"])]
    return o


def ssh_cmd(s: dict, remote: str) -> list[str]:
    return ["ssh", *_opts(s), host_label(s), remote]


def run(s: dict, remote: str, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(ssh_cmd(s, remote), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise SshError(f"ssh timed out after {timeout:.0f}s")
    except FileNotFoundError:
        raise SshError("ssh client not found — install openssh-client.")


def run_ok(s: dict, remote: str, timeout: float = 60) -> str:
    r = run(s, remote, timeout)
    if r.returncode != 0:
        raise SshError((r.stderr or r.stdout).strip() or f"ssh exited with {r.returncode}")
    return r.stdout


def remote_home(s: dict) -> str:
    return run_ok(s, 'printf %s "$HOME"').strip()


def expand(path: str, home: str) -> str:
    if path == "~":
        return home
    return home + path[1:] if path.startswith("~/") else path


def _copy(s: dict, src: str, dst: str, upload: bool, timeout: float | None = None):
    """rsync when both sides have it, otherwise scp -r."""
    remote_side = f"{host_label(s)}:{dst if upload else src}"   # scp may use SFTP: no shell quoting
    if shutil.which("rsync"):
        e = shlex.join(["ssh", *_opts(s)])
        a, b = (src.rstrip("/") + "/", remote_side + "/") if upload else (remote_side + "/", dst.rstrip("/") + "/")
        r = subprocess.run(["rsync", "-az", "-e", e, a, b], stdin=subprocess.DEVNULL, timeout=timeout)
        if r.returncode == 0:
            return
        print(f"YAAP: rsync failed (exit {r.returncode}), trying scp…", flush=True)
    a, b = (src.rstrip("/") + "/.", remote_side) if upload else (remote_side + "/.", dst)
    if not upload:
        Path(dst).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-r", *_opts(s, "-P"), a, b], stdin=subprocess.DEVNULL, timeout=timeout)
    if r.returncode != 0:
        raise SshError(f"file transfer failed (scp exit {r.returncode})")


def _copy_file(s: dict, src: Path, dst: str):
    r = subprocess.run(["scp", *_opts(s, "-P"), str(src), f"{host_label(s)}:{dst}"],
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise SshError(f"upload of {src.name} failed (scp exit {r.returncode})")


# ── test connection ──────────────────────────────────────────────────
_PROBE = ("import json, torch, ultralytics; print(json.dumps({'torch': torch.__version__, "
          "'cuda': torch.cuda.is_available(), 'gpus': [torch.cuda.get_device_name(i) "
          "for i in range(torch.cuda.device_count())], 'ultralytics': ultralytics.__version__}))")


def test_connection(s: dict) -> dict:
    validate(s)
    home = remote_home(s)
    py = shlex.quote(expand(s["python"], home))
    out = run_ok(s, f"{py} -c {shlex.quote(_PROBE)}", timeout=90)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise SshError(f"Unexpected output from remote python: {out.strip()[:300]}")


# ── remote job ───────────────────────────────────────────────────────
def remote_job_dir(s: dict, job_id: int, home: str) -> str:
    return f"{expand(s['workdir'], home).rstrip('/')}/job_{int(job_id)}"


def kill_remote(s: dict, remote_pid: int):
    """The training runs under setsid, so its pid is also the process-group id."""
    run(s, f"kill -TERM -- -{int(remote_pid)} 2>/dev/null; kill -TERM {int(remote_pid)} 2>/dev/null; true", timeout=20)


def remove_remote(s: dict, job_dir: str):
    if not re.search(r"/job_\d+$", job_dir):    # never rm -rf anything but a job folder
        raise SshError("Refusing to remove a non-job directory.")
    run(s, f"rm -rf -- {shlex.quote(job_dir)}", timeout=60)


def _remote_state(s: dict, jd: str) -> tuple[str, int | None]:
    """('done', exit_code) | ('alive', None) | ('dead', None) | ('unreachable', None)"""
    d = shlex.quote(jd)
    cmd = (f"cd {d} && if [ -f exitcode ]; then echo DONE $(cat exitcode); "
           f"elif kill -0 $(cat pid 2>/dev/null) 2>/dev/null; then echo ALIVE; else echo DEAD; fi")
    try:
        r = run(s, cmd, timeout=30)
    except SshError:
        return "unreachable", None
    if r.returncode == 255:
        return "unreachable", None
    parts = r.stdout.split()
    if parts[:1] == ["DONE"]:
        return "done", int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 1
    return ("alive" if parts[:1] == ["ALIVE"] else "dead"), None


def _stream_log(s: dict, jd: str, pid: int, out) -> str:
    """Follow the remote train.log into `out` (the job's local log) until the
    remote process is gone. Reconnects on dropped connections, resuming by byte
    offset so nothing is duplicated. Returns 'done' | 'dead'."""
    sent = 0
    unreachable_since = None
    while True:
        cmd = f"tail -c +{sent + 1} -F --pid={pid} {shlex.quote(jd + '/train.log')} 2>/dev/null"
        p = subprocess.Popen(ssh_cmd(s, cmd), stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        while True:
            chunk = os.read(p.stdout.fileno(), 65536)
            if not chunk:
                break
            out.write(chunk)
            out.flush()
            sent += len(chunk)
        p.wait()
        state, _ = _remote_state(s, jd)
        if state in ("done", "dead"):
            r = run(s, f"tail -c +{sent + 1} {shlex.quote(jd + '/train.log')}", timeout=60)  # catch-up
            if r.returncode == 0 and r.stdout:
                out.write(r.stdout.encode())
                out.flush()
            return state
        if state == "unreachable":
            unreachable_since = unreachable_since or time.time()
            if time.time() - unreachable_since > 900:
                raise SshError("Lost connection to the remote machine for over 15 minutes.")
            print("\nYAAP: connection lost, retrying…", flush=True)
        else:
            unreachable_since = None
        time.sleep(5)


def run_job(job, params: dict, db) -> tuple[Path | None, str]:
    """Upload → train detached → stream log → download. Returns (best_local_path, error).
    `job.params` already carries the settings snapshot under 'ssh'."""
    s = params["ssh"]
    validate(s)
    out = sys.stdout.buffer

    def say(msg):
        print(f"YAAP: {msg}", flush=True)

    home = remote_home(s)
    py = expand(s["python"], home)
    jd = remote_job_dir(s, job.id, home)
    params["remote_dir"] = jd
    job.params = json.dumps(params)
    db.commit()

    say(f"preparing {host_label(s)}:{jd}")
    run_ok(s, f"mkdir -p {shlex.quote(jd)}")
    say("uploading dataset…")
    _copy(s, params["local_data_dir"], f"{jd}/dataset", upload=True)
    _copy_file(s, REMOTE_SCRIPT, f"{jd}/remote_train.py")

    model = job.model_arch
    if Path(model).is_file():                # local uploaded/trained weights; aliases download remotely
        say("uploading starting weights…")
        _copy_file(s, Path(model), f"{jd}/model.pt")
        model = "model.pt"

    data = "dataset" if params["task"] == "classify" else "dataset/data.yaml"
    args = ["--data", data, "--model", model, "--task", params["task"],
            "--epochs", params["epochs"], "--imgsz", params["imgsz"], "--batch", params["batch"],
            "--patience", params["patience"], "--device", params["device"],
            "--project", "runs", "--name", "run", "--extra", json.dumps(params.get("extra") or {})]
    if params.get("is_rtdetr"):
        args.append("--rtdetr")
    train = " ".join(shlex.quote(str(x)) for x in [py, "remote_train.py", *args])
    inner = f"echo $$ > pid; {train}; echo $? > exitcode"
    # `;` not `&&` before the nohup: `a && b &` backgrounds the whole list in a subshell
    # that keeps ssh's stdout open, so ssh would not return until training ended
    launch = (f"cd {shlex.quote(jd)} || exit 1; rm -f pid exitcode; "
              f"nohup setsid sh -c {shlex.quote(inner)} > train.log 2>&1 < /dev/null & "
              f"for i in $(seq 1 20); do [ -s {shlex.quote(jd)}/pid ] && break; sleep 0.5; done; "
              f"cat {shlex.quote(jd)}/pid")
    pid = int(run_ok(s, launch).strip().splitlines()[-1])
    params["remote_pid"] = pid
    job.params = json.dumps(params)
    db.commit()
    say(f"training started on {host_label(s)} (remote pid {pid})")

    state = _stream_log(s, jd, pid, out)
    _, code = _remote_state(s, jd)
    if state != "done" or code != 0:
        return None, f"remote training failed (exit code {code if state == 'done' else 'killed'})"

    res = run(s, f"grep '^YAAP_RESULT ' {shlex.quote(jd + '/train.log')} | tail -n 1", timeout=30).stdout.strip()
    if not res:
        return None, "remote training finished but reported no result"
    rel = Path(json.loads(res.split(" ", 1)[1])["best"])
    run_name = rel.parts[0]
    say("downloading results…")
    local_run = Path(params["local_run_dir"])
    local_run.mkdir(parents=True, exist_ok=True)
    _copy(s, f"{jd}/runs/{run_name}", str(local_run), upload=False)
    best = local_run.joinpath(*rel.parts[1:])

    if s.get("cleanup"):
        say("removing remote job folder")
        remove_remote(s, jd)
    return (best if best.exists() else None), ("" if best.exists() else "best.pt missing after download")
