#!/usr/bin/env bash
# bash dev.sh [--docker]
#   default:  run the ML service locally via uv
#   --docker: run the ML service in a Docker container instead
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=5000

USE_DOCKER=""
for arg in "$@"; do
  case "$arg" in
    --docker) USE_DOCKER=1 ;;
    *) echo "Unknown option: $arg"; echo "Usage: bash dev.sh [--docker]"; exit 1 ;;
  esac
done

# ML service will save inputs and outputs for each prediction
export ML_DEBUG="true"

# Fail fast if something already holds the port (stale container, old run).
if [ -n "$(ss -ltnH "sport = :$PORT")" ]; then
  echo "Error: port $PORT is already in use. Find the holder with:"
  echo "  docker ps --filter publish=$PORT   or   ss -ltnp 'sport = :$PORT'"
  exit 1
fi

cleanup() {
  echo "Shutting down..."
  if [ -n "$ML_PID" ]; then kill "$ML_PID" 2>/dev/null || true; fi
  if [ -n "$USED_DOCKER" ]; then docker compose -f "$ROOT/docker-compose.yml" down; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM  # route Ctrl+C through the EXIT trap, exactly once

# Start ML service
if [ -n "$USE_DOCKER" ]; then
  if ! docker info &>/dev/null; then
    echo "Error: --docker given but the Docker daemon isn't reachable."
    echo "Start Docker (or drop the flag to run the ML service locally)."
    exit 1
  fi
  echo "Starting ML service via Docker..."
  docker compose -f "$ROOT/docker-compose.yml" up --build -d
  USED_DOCKER=1
else
  echo "Starting ML service locally..."
  cd "$ROOT/ml-service"
  if command -v uv &>/dev/null; then
    uv run python app.py &   # syncs .venv from uv.lock automatically
  elif [ -f .venv/bin/python ]; then
    .venv/bin/python app.py &
  else
    echo "Note: uv not found and no .venv present; trying system python3."
    echo "If this fails, install uv (https://docs.astral.sh/uv/) and rerun."
    python3 app.py &
  fi
  ML_PID=$!
  cd "$ROOT"
fi

# Wait for Flask: GET on the POST-only /transcribe returns 405 once it's up.
echo "Waiting for ML service on port $PORT..."
for i in $(seq 1 30); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/transcribe")" = "405" ]; then
    break
  fi
  if [ "$i" = "30" ]; then echo "Error: ML service never came up."; exit 1; fi
  sleep 2
done
echo "ML service ready."

npm run tauri dev
