# syntax=docker/dockerfile:1.7

# Base image with CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# ---------- Builder: compile dependency wheels ----------
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip wheel --wheel-dir=/wheels -r /app/requirements.txt

# ---------- Runtime: slim environment with CUDA libs ----------
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-index --find-links=/wheels -r /app/requirements.txt && \
    rm -rf /wheels

COPY . .

ENV MARKET_RADAR_MODEL_CACHE=/app/models \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["python", "-m", "market_radar"]
CMD ["--config", "/app/config.yaml"]
