#!/bin/sh
# Runs as the unprivileged "yaap" user — optionally pre-download model
# weights into the mounted data volume so subsequent runs work fully offline.
set -e
if [ -n "$PRELOAD_MODELS" ]; then
  echo "[yaap] preloading models: $PRELOAD_MODELS"
  python - << 'PY' || echo "[yaap] preload failed (no network?) — continuing; upload .pt files via the UI."
import os
names = os.environ.get("PRELOAD_MODELS", "").split()
if names:
    from ultralytics.utils.downloads import attempt_download_asset
    os.chdir("/app/data/models")
    for n in names:
        print("  ->", n)
        attempt_download_asset(n)
PY
fi
exec uvicorn backend.app:app --host 0.0.0.0 --port 8811
