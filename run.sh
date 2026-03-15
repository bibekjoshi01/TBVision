#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT_DIR"
BACKEND_CMD="python -m uvicorn tbvision.main:app --reload"

cd "$ROOT_DIR"

# Start backend first so static folder and services initialize before the frontend consumes them.
$BACKEND_CMD &

# Frontend setup
cd "$ROOT_DIR/frontend"
if [[ ! -d "node_modules" ]]; then
  npm install
fi

# Launch frontend dev server; frontend logs stay in foreground while backend is cleaned up by the trap.
npm run dev -- --hostname 0.0.0.0 --port 3000
