#!/bin/sh
#Bash script to start the server, preloading models if requested
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
        try:
            attempt_download_asset(n)
        except Exception as e:
            print("     failed:", e)
    # every training's dataset check needs this font (downloaded into $YOLO_CONFIG_DIR/Ultralytics,
    # here as the yaap user, so training never has to fetch it — or fail on a root-owned folder)
    try:
        from ultralytics.utils.checks import check_font
        check_font("Arial.ttf")
        print("  -> Arial.ttf")
    except Exception as e:
        print("     font preload failed:", e)
PY
fi
exec uvicorn backend.app:app --host 0.0.0.0 --port 8811
