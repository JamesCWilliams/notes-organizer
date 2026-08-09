#!/usr/bin/env bash
#bash dev.sh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT=5000

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
if docker info &>/dev/null; then
  echo "Starting ML service via Docker..."
  docker compose -f "$ROOT/docker-compose.yml" up --build -d
  USED_DOCKER=1
else
  echo "Docker not available, starting ML service directly..."
  cd "$ROOT/ml-service"
  if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
  $PYTHON app.py &
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
