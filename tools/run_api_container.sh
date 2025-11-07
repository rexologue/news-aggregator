#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   PORT=58000 CONFIG_PATH=~/market-radar/config.yaml ./tools/run_api_container.sh
#
# Description:
#   Builds (optionally) and starts the Market Radar FastAPI container. Configure the
#   runtime through the following environment variables before invoking the script:
#     IMAGE_NAME       – image tag (default: market-radar-api)
#     PORT             – host/container port (default: 8000)
#     CONFIG_PATH      – host path to the YAML config mounted read-only
#     SOURCES_PATH     – optional host path to sources JSON mounted read-only
#     MODELS_DIR       – host dir bound to /app/models for HF cache (default: ~/.cache/huggingface)
#     DO_BUILD         – 1 to (re)build the Docker image (default: 0)
#     DEV              – 1 to mount the repo as /app for live-editing (default: 0)
#     OPENROUTER_API_KEY – forwarded to the container when present

IMAGE_NAME=${IMAGE_NAME:-market-radar-api}
PORT=${PORT:-8000}
CONFIG_PATH=${CONFIG_PATH:-$(pwd)/config.yaml}
SOURCES_PATH=${SOURCES_PATH:-}
MODELS_DIR=${MODELS_DIR:-~/.cache/huggingface}
DO_BUILD=${DO_BUILD:-0}
DEV=${DEV:-0}

if ! [[ ${PORT} =~ ^[0-9]+$ ]]; then
  echo "Error: PORT must be numeric (got '${PORT}')." >&2
  exit 1
fi

resolve_path() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

CONFIG_PATH_ABS=$(resolve_path "${CONFIG_PATH}")
if [[ ! -f ${CONFIG_PATH_ABS} ]]; then
  echo "Error: configuration file not found at '${CONFIG_PATH_ABS}'." >&2
  exit 1
fi
CONFIG_DIR_ABS=$(dirname "${CONFIG_PATH_ABS}")

if [[ -n ${SOURCES_PATH} ]]; then
  SOURCES_PATH_ABS=$(resolve_path "${SOURCES_PATH}")
  if [[ ! -f ${SOURCES_PATH_ABS} ]]; then
    echo "Error: sources file not found at '${SOURCES_PATH_ABS}'." >&2
    exit 1
  fi
  SOURCES_DIR_ABS=$(dirname "${SOURCES_PATH_ABS}")
fi

MODELS_DIR_ABS=$(resolve_path "${MODELS_DIR}")
mkdir -p "${MODELS_DIR_ABS}"

if [[ ${DO_BUILD} -eq 1 ]]; then
  echo "Building image '${IMAGE_NAME}' with Docker BuildKit..."
  DOCKER_BUILDKIT=1 docker build -t "${IMAGE_NAME}" .
fi

RUN_ARGS=(
  docker run --rm
  -p "${PORT}:${PORT}"
  -e "PORT=${PORT}"
  -e "MARKET_RADAR_MODEL_CACHE=/app/models"
  -e "HF_HOME=/app/models"
  -e "TRANSFORMERS_CACHE=/app/models"
  -e "SENTENCE_TRANSFORMERS_HOME=/app/models"
  -v "${MODELS_DIR_ABS}:/app/models"
  -v "${CONFIG_DIR_ABS}:${CONFIG_DIR_ABS}:ro"
)

if [[ -n ${SOURCES_PATH:-} ]]; then
  RUN_ARGS+=( -v "${SOURCES_DIR_ABS}:${SOURCES_DIR_ABS}:ro" )
fi

if [[ -n ${OPENROUTER_API_KEY:-} ]]; then
  RUN_ARGS+=( -e "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" )
fi

if [[ ${DEV} -eq 1 ]]; then
  RUN_ARGS+=( -v "$(pwd):/app:ro" )
fi

RUN_ARGS+=( "${IMAGE_NAME}" "--config" "${CONFIG_PATH_ABS}" )

echo "Executing: ${RUN_ARGS[*]}"
exec "${RUN_ARGS[@]}"
