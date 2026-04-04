#!/usr/bin/env bash
# Launch ant-mnist viz backend + React dashboard together.
# Usage: ./dev.sh [data_dir] [ws_port] [http_port]

DATA_DIR="${1:-data}"
WS_PORT="${2:-8091}"
HTTP_PORT="${3:-8090}"
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
echo "=== Starting backend (WS :${WS_PORT}, HTTP :${HTTP_PORT})..."
cargo run --release --bin ant-mnist-viz -- "$DATA_DIR" "$WS_PORT" "$HTTP_PORT" &
BACKEND_PID=$!

# Wait for WebSocket to be ready (check if port is listening)
echo "=== Waiting for backend..."
for i in $(seq 1 30); do
  (echo > /dev/tcp/localhost/${WS_PORT}) 2>/dev/null && break
  sleep 0.5
done
sleep 1
echo "=== Backend ready."

# Start frontend
echo "=== Starting dashboard..."
cd "$SCRIPT_DIR/dashboard"
npx vite --port 5173 --strictPort &
FRONTEND_PID=$!

echo ""
echo "=== ANT-MNIST Control Room v3 ==="
echo "  Dashboard:  http://localhost:5173"
echo "  WebSocket:  ws://localhost:${WS_PORT}"
echo "  HTTP:       http://localhost:${HTTP_PORT}"
echo "  Ctrl+C to stop both."
echo ""

wait
