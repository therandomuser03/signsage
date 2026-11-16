#!/usr/bin/env bash
# shortcut: relies on desktop terminal emulator and GUI display; use tmux if running headless or over SSH
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

# Check backend virtual environment (.venv or venv)
VENV_PATH=""
if [[ -d "$BACKEND_DIR/.venv" ]]; then
  VENV_PATH="$BACKEND_DIR/.venv"
elif [[ -d "$BACKEND_DIR/venv" ]]; then
  VENV_PATH="$BACKEND_DIR/venv"
else
  echo "Error: Backend virtual environment not found in $BACKEND_DIR (.venv or venv)." >&2
  echo "Run: cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# Check frontend node_modules
if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Error: Frontend node_modules not found in $FRONTEND_DIR." >&2
  echo "Run: cd frontend && npm install" >&2
  exit 1
fi

launch_terminal() {
  local title="$1" dir="$2" cmd="$3"

  if command -v ptyxis >/dev/null 2>&1; then
    ptyxis --new-window --working-directory="$dir" --title="$title" -- bash -c "$cmd; exec bash" &
  elif command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" --working-directory="$dir" -- bash -c "$cmd; exec bash" &
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -T "$title" -e bash -c "cd '$dir' && $cmd; exec bash" &
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "$title" -e bash -c "cd '$dir' && $cmd; exec bash" &
  else
    echo "Error: No supported terminal emulator found (tried ptyxis, gnome-terminal, x-terminal-emulator, xterm)." >&2
    exit 1
  fi
}

echo "Pre-flight checks passed."
echo "Opening Backend terminal..."
launch_terminal "SignSage Backend" "$BACKEND_DIR" "source '$VENV_PATH/bin/activate' && uvicorn app.main:app --reload"

echo "Opening Frontend terminal..."
launch_terminal "SignSage Frontend" "$FRONTEND_DIR" "npm run dev"

echo "Both servers launched in separate terminals."
