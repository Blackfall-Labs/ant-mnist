#!/usr/bin/env bash
# Launch ant-mnist viz backend + React dashboard together.
# Usage: ./dev.sh [data_dir] [backend_port]
#   data_dir     — path to MNIST data (default: data)
#   backend_port — Rust server port (default: 8090)

DATA_DIR="${1:-data}"
BACKEND_PORT="${2:-8090}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "Shutting down..."
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null
  wait 2>/dev/null
  echo "Done."
}
trap cleanup EXIT INT TERM

# Build backend
echo "=== Building backend..."
cargo build --release --bin ant-mnist-viz 2>&1
if [[ $? -ne 0 ]]; then
  echo "Backend build failed."
  exit 1
fi

# Start backend
echo "=== Starting backend on :${BACKEND_PORT}..."
cargo run --release --bin ant-mnist-viz -- "$DATA_DIR" "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait for backend to be ready (up to 15s)
echo "=== Waiting for backend..."
for i in $(seq 1 30); do
  curl -s "http://localhost:${BACKEND_PORT}/api/state" > /dev/null 2>&1 && break
  sleep 0.5
done

if ! curl -s "http://localhost:${BACKEND_PORT}/api/state" > /dev/null 2>&1; then
  echo "Backend failed to start."
  exit 1
fi
echo "=== Backend ready."

# Start frontend
echo "=== Starting dashboard..."
cd "$SCRIPT_DIR/dashboard"
npx vite --port 5173 --strictPort &
FRONTEND_PID=$!

echo ""
echo "=== ANT-MNIST Control Room ==="
echo "  Dashboard:  http://localhost:5173"
echo "  Backend:    http://localhost:${BACKEND_PORT}"
echo "  Ctrl+C to stop both."
echo ""

wait
