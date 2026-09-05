#!/bin/bash
# Runs detached in the background (started once by launch.sh). Polls for the
# "browser actually closed" marker the backend drops at data/.leave_at (see
# POST /api/system/leaving in backend/routers/models.py) and, once it's older
# than GRACE seconds with no newer activity clearing it, stops the container.
# A reload or quick reopen clears the marker (POST /api/system/here on page
# load) before the grace period is up, so it never fires on those.
cd "$(dirname "$0")"
GRACE=60
MARKER="data/.leave_at"

while true; do
  sleep 15
  if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
    exit 0   # container already stopped some other way — nothing left to watch
  fi
  if [ -f "$MARKER" ]; then
    left_at=$(cat "$MARKER" 2>/dev/null || echo 0)
    now=$(date +%s)
    age=$(( now - left_at ))
    if [ "$age" -ge "$GRACE" ]; then
      docker compose down
      rm -f "$MARKER"
      exit 0
    fi
  fi
done
