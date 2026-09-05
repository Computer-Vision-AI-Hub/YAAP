#!/bin/bash

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
