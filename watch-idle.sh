#!/bin/bash
#Watchdog script to stop the docker compose stack if it has been idle for a while. 
#The stack is considered idle if the file data/.leave_at exists and is older than GRACE seconds. 
#The file is created by the leave.sh script when the user leaves the stack.

cd "$(dirname "$0")"  #cd to watch-idle.sh directory so that the script can be run from anywhere
GRACE=60
MARKER="data/.leave_at"

while true; do
  sleep 15
  #Check if the container is running. If not, exit the script.
  if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
    exit 0   # container already stopped some other way
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
