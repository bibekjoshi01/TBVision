#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_CMD="python tbvision/run_backend.py"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" && -e /proc/$BACKEND_PID ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

cd "$ROOT_DIR"

# Start backend first so static folder and services initialize before the frontend consumes them.
$BACKEND_CMD &
BACKEND_PID=$!

# Frontend setup
cd "$ROOT_DIR/frontend"
if [[ ! -d "node_modules" ]]; then
  npm install
fi

# Launch frontend dev server; frontend logs stay in foreground while backend is cleaned up by the trap.
npm run dev -- --hostname 0.0.0.0 --port 3000
