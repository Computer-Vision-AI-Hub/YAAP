#!/bin/bash
# YAAP portable launcher.
# Pulls the published computervisionaihub/yaap image, auto-detects GPU vs CPU, and installs its own
# desktop shortcut the first time it runs.
#
#
# Overrides (optional):
#   YAAP_MODE=cpu  bash yaap-launch.sh   # force CPU even if a GPU exists
#   YAAP_MODE=gpu  bash yaap-launch.sh   # force GPU (skip detection)
#   YAAP_IDLE_SHUTDOWN=0 bash yaap-launch.sh   # never auto-stop when the tab closes
set -e

TEMPLATE_VERSION=3

YAAP_HOME="$HOME/YAAP"
SELF="$YAAP_HOME/yaap-launch.sh"
COMPOSE_FILE="$YAAP_HOME/docker-compose.yml"
STAMP_FILE="$YAAP_HOME/.compose-stamp"
WATCHER="$YAAP_HOME/watch-idle.sh"
LOG_FILE="$YAAP_HOME/.launch.log"
URL="http://127.0.0.1:8811"

# Double-clicking a .desktop icon gives a bare-bones environment, so make sure
# the usual locations of docker / nvidia-smi are on PATH.
export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"

# Model weights go in data/models.
mkdir -p "$YAAP_HOME/data/models" "$YAAP_HOME/seg_models"
# To let container write into the mounted folders.
chmod -R a+rwX "$YAAP_HOME/data" "$YAAP_HOME/seg_models" 2>/dev/null || true

# Logs.
exec > >(tee -a "$LOG_FILE") 2>&1
echo "── $(date) ──"
notify() { command -v notify-send >/dev/null 2>&1 && notify-send "YAAP" "$1" 2>/dev/null || true; }
trap 'notify "Failed to start — see '"$LOG_FILE"' for details"' ERR

if ! command -v docker >/dev/null 2>&1; then
  echo "[yaap] Docker is not installed or not on PATH."
  notify "Docker not found — install Docker first."
  exit 1
fi

notify "Starting… first run can take a few minutes (downloading the image)."

# Copy this launcher into YAAP_HOME so the desktop shortcut can always find it, and
# make it executable.
if [ "$(readlink -f "$0")" != "$(readlink -f "$SELF" 2>/dev/null || echo "")" ]; then
  cp "$0" "$SELF"
  chmod +x "$SELF"
fi

# GPU detection 
detect_mode() {
  if [ "$YAAP_MODE" = "cpu" ] || [ "$YAAP_MODE" = "gpu" ]; then
    echo "$YAAP_MODE"; return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo cpu; return
  fi
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo gpu
  else
    echo "[yaap] NVIDIA GPU found, but Docker can't use it — install the NVIDIA Container Toolkit:" >&2
    echo "        sudo apt install nvidia-container-toolkit" >&2
    echo "        sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" >&2
    notify "GPU found but Docker can't use it — running on CPU. See $LOG_FILE."
    echo cpu
  fi
}

write_compose() {
  local mode="$1"
  if [ "$mode" = "gpu" ]; then
    cat > "$COMPOSE_FILE" << EOF
services:
  yaap:
    image: computervisionaihub/yaap:latest
    container_name: yaap
    ports:
      - "127.0.0.1:8811:8811"
    volumes:
      - $YAAP_HOME/data:/app/data
      - $YAAP_HOME/seg_models:/app/seg_models
    restart: unless-stopped
    shm_size: "8gb"
    environment:
      PRELOAD_MODELS: "yolo11n.pt"
      YOLO_CONFIG_DIR: /tmp/Ultralytics
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
  else
    cat > "$COMPOSE_FILE" << EOF
services:
  yaap:
    image: computervisionaihub/yaap:cpu
    container_name: yaap
    ports:
      - "127.0.0.1:8811:8811"
    volumes:
      - $YAAP_HOME/data:/app/data
      - $YAAP_HOME/seg_models:/app/seg_models
    restart: unless-stopped
    environment:
      PRELOAD_MODELS: "yolo11n.pt"
      YOLO_CONFIG_DIR: /tmp/Ultralytics
EOF
  fi
  echo "$mode v$TEMPLATE_VERSION" > "$STAMP_FILE"
  echo "[yaap] wrote $COMPOSE_FILE ($mode mode)"
}

MODE=$(detect_mode)
echo "[yaap] detected mode: $MODE"

# Compose regereraion if the template version changed.
WANT_STAMP="$MODE v$TEMPLATE_VERSION"
HAVE_STAMP=$(cat "$STAMP_FILE" 2>/dev/null || echo "")
if [ ! -f "$COMPOSE_FILE" ] || [ "$HAVE_STAMP" != "$WANT_STAMP" ]; then
  [ -f "$COMPOSE_FILE" ] && cp "$COMPOSE_FILE" "$COMPOSE_FILE.bak"
  write_compose "$MODE"
fi

# ── Idle-shutdown watcher 
cat > "$WATCHER.tmp" << 'EOF'
#!/bin/bash
export PATH="$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
cd "$(dirname "$0")"
GRACE=120
MARKER="data/.leave_at"
log() { echo "[watcher $(date '+%F %T')] $*" >> .launch.log; }
while true; do
  sleep 15
  if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
    exit 0
  fi
  [ -f "$MARKER" ] || continue
  left_at=$(cat "$MARKER" 2>/dev/null || echo 0)
  case "$left_at" in ''|*[!0-9]*) rm -f "$MARKER"; continue ;; esac
  now=$(date +%s)
  [ $(( now - left_at )) -ge "$GRACE" ] || continue
  # Any real request after the "leaving" signal means the page is still in use.
  if docker logs --since "$left_at" yaap 2>&1 \
      | grep -E '"(GET|POST|PUT|PATCH|DELETE) ' \
      | grep -v -E '/api/system/(leaving|here)' | grep -q .; then
    log "activity after leave signal — staying up"
    rm -f "$MARKER"
    continue
  fi
  log "no activity for ${GRACE}s after page closed — stopping YAAP"
  docker compose down
  rm -f "$MARKER"
  exit 0
done
EOF
chmod +x "$WATCHER.tmp"
mv -f "$WATCHER.tmp" "$WATCHER"

# Desktop shortcut — YAAP logo from the website, generic icon if offline.
DESKTOP_FILE="$HOME/.local/share/applications/yaap.desktop"
LOGO="$YAAP_HOME/yaap-logo.png"
if [ ! -s "$LOGO" ]; then
  curl -fsSL -m 20 https://computer-vision-ai-hub.github.io/YAAP/img/yaap-logo.png -o "$LOGO" 2>/dev/null || rm -f "$LOGO"
fi
ICON="applications-graphics"
[ -s "$LOGO" ] && ICON="$LOGO"
# Shortcuts made by older versions of this script get the logo too.
for f in "$DESKTOP_FILE" "$HOME/Desktop/yaap.desktop"; do
  if [ -f "$f" ] && [ "$ICON" != "applications-graphics" ]; then
    sed -i "s|^Icon=applications-graphics$|Icon=$ICON|" "$f"
  fi
done
if [ ! -f "$DESKTOP_FILE" ]; then
  mkdir -p "$HOME/.local/share/applications"
  cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=YAAP
Comment=Yet Another Annotation Platform
Exec=bash -lc "$SELF"
Icon=$ICON
Terminal=false
Categories=Development;Graphics;
EOF
  if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$HOME/Desktop/yaap.desktop"
    chmod +x "$HOME/Desktop/yaap.desktop"
    command -v gio >/dev/null 2>&1 && gio set "$HOME/Desktop/yaap.desktop" "metadata::trusted" true 2>/dev/null || true
  fi
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
  echo "[yaap] installed desktop shortcut — find YAAP in your app launcher from now on"
fi

# Start
cd "$YAAP_HOME"
# Compose recreates the container automatically if the file changed.
if ! docker compose up -d; then
  if [ "$MODE" = "gpu" ] && [ -z "$YAAP_MODE" ]; then
    echo "[yaap] GPU start failed — falling back to CPU mode."
    notify "GPU start failed — falling back to CPU. See $LOG_FILE."
    cp "$COMPOSE_FILE" "$COMPOSE_FILE.bak"
    MODE=cpu
    write_compose cpu
    docker compose up -d
  else
    false   # trigger the ERR trap
  fi
fi
rm -f data/.leave_at

# GPU detection log.
if [ "$MODE" = "gpu" ]; then
  if docker exec yaap nvidia-smi -L 2>/dev/null; then
    echo "[yaap] GPU visible inside the container."
  else
    echo "[yaap] WARNING: GPU not visible inside the container."
  fi
fi

# First startup.
echo "Waiting for YAAP..."
for _ in $(seq 1 600); do
  if curl -s -o /dev/null -w "%{http_code}" "$URL/" 2>/dev/null | grep -q 200; then
    break
  fi
  if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
    echo "[yaap] container stopped — last log lines:"
    docker compose logs --tail 40
    false
  fi
  sleep 1
done

# Replace any old watcher, then start the
# current one unless idle shutdown is disabled.
OLD_PID=$(cat "$YAAP_HOME/.watcher.pid" 2>/dev/null || echo "")
[ -n "$OLD_PID" ] && kill "$OLD_PID" 2>/dev/null || true
rm -f "$YAAP_HOME/.watcher.pid"
if [ "${YAAP_IDLE_SHUTDOWN:-1}" != "0" ]; then
  nohup "$WATCHER" >/dev/null 2>&1 &
  echo $! > "$YAAP_HOME/.watcher.pid"
else
  echo "[yaap] idle shutdown disabled — stop with: cd ~/YAAP && docker compose down"
fi

notify "Ready ($MODE mode) — opening in your browser."
xdg-open "$URL" 2>/dev/null \
  || open "$URL" 2>/dev/null \
  || echo "YAAP is up — open $URL in your browser."
