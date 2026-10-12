#!/bin/bash
# YAAP manual install — CPU or GPU.
#
#   docker pull computervisionaihub/yaap:gpu      (or :cpu)
#
#   curl -fsSL https://raw.githubusercontent.com/Computer-Vision-AI-Hub/YAAP/main/docs/yaap-docker-install.sh | bash -s gpu
#   curl -fsSL https://raw.githubusercontent.com/Computer-Vision-AI-Hub/YAAP/main/docs/yaap-docker-install.sh | bash -s cpu
#
# Pulls the image, creates the "yaap" container, installs a desktop icon and
# starts YAAP. From then on the icon starts the container and opens the
# browser; right-click the icon → "Stop YAAP" to stop it.
set -e

MODE="$1"
case "$MODE" in
  gpu|cpu) ;;
  *) echo "Usage: bash yaap-docker-install.sh gpu|cpu"; exit 1 ;;
esac

YAAP_HOME="$HOME/YAAP"
IMAGE="computervisionaihub/yaap:$MODE"
START="$YAAP_HOME/yaap-start.sh"
APPS="$HOME/.local/share/applications"

command -v docker >/dev/null 2>&1 || { echo "Docker is not installed."; exit 1; }

if [ "$MODE" = gpu ] && ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  echo "Docker can't use the GPU (NVIDIA Container Toolkit missing). Install it with:"
  echo "  sudo apt install nvidia-container-toolkit"
  echo "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  echo "…or install the CPU version instead: bash -s cpu"
  exit 1
fi

mkdir -p "$YAAP_HOME/data/models" "$YAAP_HOME/seg_models" "$APPS"

# Use the image the user already pulled; only pull if it isn't there yet.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[yaap] using $IMAGE"
else
  echo "[yaap] $IMAGE not found locally — pulling it (several GB)…"
  docker pull "$IMAGE"
fi

if docker container inspect yaap >/dev/null 2>&1; then
  echo "[yaap] replacing the existing yaap container (data in $YAAP_HOME is kept)"
  docker rm -f yaap >/dev/null
fi

GPU_FLAGS=()
[ "$MODE" = gpu ] && GPU_FLAGS=(--gpus all)
docker create --name yaap "${GPU_FLAGS[@]}" \
  --shm-size=8gb \
  -p 127.0.0.1:8811:8811 \
  -v "$YAAP_HOME/data:/app/data" \
  -v "$YAAP_HOME/seg_models:/app/seg_models" \
  -e PRELOAD_MODELS="yolo11n.pt" \
  "$IMAGE" >/dev/null
echo "[yaap] created container ($MODE)"

# Start/stop script the icon runs (avoids .desktop quoting limits).
cat > "$START" << 'EOF'
#!/bin/bash
# Started by the YAAP desktop icon. "yaap-start.sh stop" stops it.
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin:/snap/bin"
LOG="$HOME/YAAP/.start.log"
URL="http://127.0.0.1:8811"
exec > >(tee -a "$LOG") 2>&1
echo "── $(date) ──"
notify() { command -v notify-send >/dev/null 2>&1 && notify-send "YAAP" "$1" 2>/dev/null || true; }

if [ "$1" = stop ]; then
  docker stop yaap && notify "Stopped."
  exit 0
fi

if ! docker start yaap; then
  notify "Could not start — see $LOG"
  exit 1
fi
notify "Starting…"
for _ in $(seq 1 180); do
  if curl -sf "$URL/" >/dev/null 2>&1; then
    xdg-open "$URL" 2>/dev/null || echo "Open $URL in your browser."
    exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' yaap 2>/dev/null)" != true ]; then
    docker logs --tail 40 yaap
    notify "YAAP stopped while starting — see $LOG"
    exit 1
  fi
  sleep 1
done
notify "YAAP is taking long to start — see $LOG"
exit 1
EOF
chmod +x "$START"

# Use the app's own logo if the image has it.
ICON="applications-graphics"
if docker run --rm --entrypoint cat "$IMAGE" /app/frontend/img/yaap-logo.png \
     > "$YAAP_HOME/yaap-logo.png" 2>/dev/null && [ -s "$YAAP_HOME/yaap-logo.png" ]; then
  ICON="$YAAP_HOME/yaap-logo.png"
fi

cat > "$APPS/yaap.desktop" << EOF
[Desktop Entry]
Type=Application
Name=YAAP
Comment=Yet Another Annotation Platform ($MODE)
Exec=bash "$START"
Icon=$ICON
Terminal=false
Categories=Development;Graphics;
Actions=stop;

[Desktop Action stop]
Name=Stop YAAP
Exec=bash "$START" stop
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" 2>/dev/null || true

DESKTOP_DIR=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
if [ -d "$DESKTOP_DIR" ]; then
  cp "$APPS/yaap.desktop" "$DESKTOP_DIR/yaap.desktop"
  chmod +x "$DESKTOP_DIR/yaap.desktop"
  command -v gio >/dev/null 2>&1 && gio set "$DESKTOP_DIR/yaap.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "[yaap] installed the YAAP icon (desktop + app menu)"

bash "$START"
