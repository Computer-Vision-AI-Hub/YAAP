#!/bin/bash
# YAAP portable launcher — works on any Linux device with Docker installed,
# no repo checkout needed. Pulls the published lukasiktar/yaap image, keeps
# data under ~/YAAP, auto-detects GPU vs CPU, and installs its own desktop
# shortcut the first time it runs.
#
# Setup: copy just this one file anywhere and run it once
# (`bash yaap-portable-launch.sh`) — everything else is generated.
set -e

YAAP_HOME="$HOME/YAAP"
SELF="$YAAP_HOME/yaap-launch.sh"
COMPOSE_FILE="$YAAP_HOME/docker-compose.yml"
WATCHER="$YAAP_HOME/watch-idle.sh"
LOG_FILE="$YAAP_HOME/.launch.log"
mkdir -p "$YAAP_HOME/data" "$YAAP_HOME/seg_models"

# Double-clicking a .desktop icon runs this with a bare-bones environment —
# no .bashrc/.profile sourced, so PATH may be missing wherever docker lives
# even though it works fine from a terminal. Log everything (so a silent
# failure is debuggable when there's no visible terminal) while still
# printing normally when run interactively — tee, not a plain redirect.
exec > >(tee -a "$LOG_FILE") 2>&1
echo "── $(date) ──"
notify() { command -v notify-send >/dev/null 2>&1 && notify-send "YAAP" "$1" 2>/dev/null; }
trap 'notify "Failed to start — see '"$LOG_FILE"' for details"' ERR
notify "Starting… first run can take a few minutes (downloading the image)."

# Copy myself into a stable location so the desktop shortcut always has a
# fixed target, regardless of where this file was first downloaded to.
if [ "$(readlink -f "$0")" != "$(readlink -f "$SELF" 2>/dev/null || echo "")" ]; then
  cp "$0" "$SELF"
  chmod +x "$SELF"
fi

# Generate the compose file once — never overwrite later manual edits.
if [ ! -f "$COMPOSE_FILE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    IMAGE="lukasiktar/yaap:latest"
    cat > "$COMPOSE_FILE" << EOF
services:
  yaap:
    image: $IMAGE
    container_name: yaap
    ports:
      - "127.0.0.1:8811:8811"
    volumes:
      - $YAAP_HOME/data:/app/data
      - $YAAP_HOME/seg_models:/app/seg_models
    restart: unless-stopped
    shm_size: "8gb"
    environment:
      PRELOAD_MODELS: "sam3.pt yolo11n.pt"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
  else
    IMAGE="lukasiktar/yaap:cpu"
    cat > "$COMPOSE_FILE" << EOF
services:
  yaap:
    image: $IMAGE
    container_name: yaap
    ports:
      - "127.0.0.1:8811:8811"
    volumes:
      - $YAAP_HOME/data:/app/data
      - $YAAP_HOME/seg_models:/app/seg_models
    restart: unless-stopped
    environment:
      PRELOAD_MODELS: "sam3.pt"
EOF
  fi
  echo "[yaap] first run — wrote $COMPOSE_FILE ($IMAGE)"
fi

# Generate the idle-shutdown watcher once (same idea as the repo's
# watch-idle.sh — relies on the backend's /api/system/leaving,here endpoints,
# which are already baked into the image).
if [ ! -f "$WATCHER" ]; then
  cat > "$WATCHER" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
GRACE=60
MARKER="data/.leave_at"
while true; do
  sleep 15
  if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
    exit 0
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
EOF
  chmod +x "$WATCHER"
fi

# Install the desktop shortcut once.
DESKTOP_FILE="$HOME/.local/share/applications/yaap.desktop"
if [ ! -f "$DESKTOP_FILE" ]; then
  mkdir -p "$HOME/.local/share/applications"
  cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=YAAP
Comment=Yet Another Annotation Platform
Exec=bash -lc "$SELF"
Icon=applications-graphics
Terminal=false
Categories=Development;Graphics;
EOF
  if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$HOME/Desktop/yaap.desktop"
    chmod +x "$HOME/Desktop/yaap.desktop"
    command -v gio >/dev/null 2>&1 && gio set "$HOME/Desktop/yaap.desktop" "metadata::trusted" yes 2>/dev/null
  fi
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$HOME/.local/share/applications" 2>/dev/null
  echo "[yaap] installed desktop shortcut — find YAAP in your app launcher from now on"
fi

# Start (docker compose pulls the image automatically on first run only).
cd "$YAAP_HOME"
docker compose up -d
rm -f data/.leave_at

echo "Waiting for YAAP..."
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8811/ 2>/dev/null | grep -q 200; do
  sleep 1
done

if ! { [ -f "$YAAP_HOME/.watcher.pid" ] && kill -0 "$(cat "$YAAP_HOME/.watcher.pid" 2>/dev/null)" 2>/dev/null; }; then
  nohup "$WATCHER" >/dev/null 2>&1 &
  echo $! > "$YAAP_HOME/.watcher.pid"
fi

notify "Ready — opening in your browser."
xdg-open http://127.0.0.1:8811 2>/dev/null \
  || open http://127.0.0.1:8811 2>/dev/null \
  || echo "YAAP is up — open http://127.0.0.1:8811 in your browser."
