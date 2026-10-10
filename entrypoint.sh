#!/bin/sh
set -e
chown -R yaap:yaap /app/data /app/seg_models 2>/dev/null || true
exec su -s /bin/sh yaap -c /app/start.sh
