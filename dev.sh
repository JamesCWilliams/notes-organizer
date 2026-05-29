#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "Shutting down..."
  if [ -n "$ML_PID" ]; then
    kill "$ML_PID" 2>/dev/null || true
  fi
  if [ "$USED_DOCKER" = "1" ]; then
    docker compose -f "$ROOT/docker-compose.yml" down
  fi
}
trap cleanup EXIT INT TERM

# Start ML service
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  echo "Starting ML service via Docker..."
  docker compose -f "$ROOT/docker-compose.yml" up --build -d
  USED_DOCKER=1
else
  echo "Docker not available, starting ML service directly..."
  cd "$ROOT/ml-service"
  if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
  $PYTHON app.py &
  ML_PID=$!
  cd "$ROOT"
fi

# Wait for ML service to be ready
echo "Waiting for ML service on port 5000..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:5000 &>/dev/null || curl -sf -o /dev/null -w "%{http_code}" http://localhost:5000/transcribe 2>/dev/null | grep -q "405"; then
    break
  fi
  sleep 2
done
echo "ML service ready."

# Start Tauri dev server
cd "$ROOT"
npm run tauri dev
