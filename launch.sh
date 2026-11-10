#!/bin/bash
# Starts YAAP using the already-built image and opens it in the default browser.
set -e
cd "$(dirname "$0")"

docker compose up -d
#Remove the .leave_at file if it exists, so that the auto-shutdown watcher doesn't shut down YAAP immediately after starting it.
rm -f data/.leave_at   

echo "Waiting for YAAP..."
#Write http code to terminal when YAAP is up
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8811/ 2>/dev/null | grep -q 200; do
  sleep 1
done

# Start the auto-shutdown watcher 
if ! { [ -f .watcher.pid ] && kill -0 "$(cat .watcher.pid 2>/dev/null)" 2>/dev/null; }; then
  nohup ./watch-idle.sh >/dev/null 2>&1 &
  echo $! > .watcher.pid
fi

#Open the YAAP web interface in the default browser, if possible. If not, print a message to the terminal.
xdg-open http://127.0.0.1:8811 2>/dev/null \
  || open http://127.0.0.1:8811 2>/dev/null \
  || echo "YAAP is up — open http://127.0.0.1:8811 in your browser."
