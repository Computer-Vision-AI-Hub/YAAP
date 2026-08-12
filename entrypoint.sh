#!/bin/sh
# YAAP entrypoint — runs as root just long enough to fix ownership on the
# bind-mounted volumes (so files the container creates belong to the host
# user, not root), then drops to the unprivileged "yaap" user for everything
# else, including the actual server process.
set -e
chown -R yaap:yaap /app/data /app/seg_models 2>/dev/null || true
exec su -s /bin/sh yaap -c /app/start.sh
