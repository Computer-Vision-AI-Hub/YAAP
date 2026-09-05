#!/bin/bash
# Starts YAAP using the already-built image (no rebuild) and opens it in the
# default browser once it's actually responding.
set -e
cd "$(dirname "$0")"

docker compose up -d
rm -f data/.leave_at   # stale marker from a previous run shouldn't trigger an immediate shutdown

echo "Waiting for YAAP..."
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8811/ 2>/dev/null | grep -q 200; do
  sleep 1
done

# start the auto-shutdown watcher once (it stays running across page reloads
# and even browser restarts — only actually closing the app for good triggers it)
if ! { [ -f .watcher.pid ] && kill -0 "$(cat .watcher.pid 2>/dev/null)" 2>/dev/null; }; then
  nohup ./watch-idle.sh >/dev/null 2>&1 &
  echo $! > .watcher.pid
fi

xdg-open http://127.0.0.1:8811 2>/dev/null \
  || open http://127.0.0.1:8811 2>/dev/null \
  || echo "YAAP is up — open http://127.0.0.1:8811 in your browser."
