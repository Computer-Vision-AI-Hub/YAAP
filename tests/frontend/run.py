#!/usr/bin/env python3
"""Run the frontend tests: every tests/frontend/*.test.html in headless Chrome.

    python3 tests/frontend/run.py            # all tests
    python3 tests/frontend/run.py annotate   # only files whose name contains "annotate"

Needs Chrome/Chromium on the host (set CHROME=/path/to/binary if it isn't on
PATH). Serves the repo root on a random localhost port with /static/ mapped to
frontend/, the same way the FastAPI app serves it. Exits non-zero on any failure.
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.startswith("/static/"):
            path = "/frontend/" + path[len("/static/"):]
        return super().translate_path(path)

    def log_message(self, *args):
        pass


def find_chrome():
    for name in (os.environ.get("CHROME"), "google-chrome", "google-chrome-stable",
                 "chromium", "chromium-browser"):
        if name and shutil.which(name):
            return shutil.which(name)
    sys.exit("Chrome/Chromium not found — install it or set CHROME=/path/to/chrome")


def run_one(chrome, url):
    out = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1280,800",
         "--virtual-time-budget=30000", "--dump-dom", url],
        capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"<title>RESULT (.*?)</title>", out, re.S)
    if not m:
        return None
    return json.loads(html.unescape(m.group(1)))


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in HERE.glob("*.test.html") if pattern in p.name)
    if not files:
        sys.exit(f"no tests match '{pattern}'")
    chrome = find_chrome()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    total_pass = total_fail = 0
    for f in files:
        res = run_one(chrome, f"{base}/tests/frontend/{f.name}")
        if res is None:
            print(f"✗ {f.name}: no result (page crashed or never finished)")
            total_fail += 1
            continue
        total_pass += len(res["passed"])
        total_fail += len(res["failed"])
        mark = "✓" if not res["failed"] else "✗"
        print(f"{mark} {f.name}: {len(res['passed'])} passed, {len(res['failed'])} failed")
        for name in res["failed"]:
            print(f"    FAIL {name}")
        if res["unmatched"]:
            print(f"    (unmocked requests: {', '.join(sorted(set(res['unmatched'])))})")
    server.shutdown()
    print(f"\n{total_pass} passed, {total_fail} failed")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
