# ─────────────────────────────────────────────────────────────────────
# YAAP — Yet Another Annotation Platform
#
# Two build targets:
#   gpu (default)  docker build -t yaap .          ← last stage wins
#   cpu            docker build -t yaap:cpu --target cpu .
#
# All state (images, labels, models, runs, DB) is bind-mounted at /app/data,
# so nothing ever leaves the host. See docker-compose.yml.
# ─────────────────────────────────────────────────────────────────────

# ── CPU image ────────────────────────────────────────────────────────
FROM python:3.11-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/data/models \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/home/yaap

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user — UID/GID default to 1000 but can be overridden at build time
# to match your host user, so bind-mounted files stay owned by you, not root.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} yaap && useradd -m -u ${UID} -g ${GID} -s /bin/sh yaap

WORKDIR /app
COPY requirements.txt .
# CPU-only torch wheels keep the image several GB smaller
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY tests ./tests
COPY run.py entrypoint.sh start.sh pytest.ini ./
RUN chmod +x entrypoint.sh start.sh && chown -R yaap:yaap /app

EXPOSE 8811
VOLUME ["/app/data"]
CMD ["./entrypoint.sh"]

# ── GPU image (NVIDIA) — default build target ───────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_CONFIG_DIR=/app/data/models \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/home/yaap

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Non-root user — UID/GID default to 1000 but can be overridden at build time
# to match your host user, so bind-mounted files stay owned by you, not root.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} yaap && useradd -m -u ${UID} -g ${GID} -s /bin/sh yaap

WORKDIR /app
COPY requirements.txt .
# CUDA 12.4 torch wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY tests ./tests
COPY run.py entrypoint.sh start.sh pytest.ini ./
RUN chmod +x entrypoint.sh start.sh && chown -R yaap:yaap /app

EXPOSE 8811
VOLUME ["/app/data"]
CMD ["./entrypoint.sh"]
