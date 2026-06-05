#!/bin/bash
# Dev launcher: runs the standalone app, which auto-spawns the Python sidecar.
# (In the shipped .app the sidecar + a private python live inside the bundle.)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ACE15_SPAWN_SIDECAR=1
export ACE15_PYTHON="$ROOT/.venv/bin/python"
export ACE15_SIDECAR="$ROOT/sidecar/server.py"
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Kill any orphaned sidecar from a previous crash/force-quit. A force-quit (SIGKILL)
# does NOT propagate to the spawned child, so the old sidecar (possibly pre-fix code,
# still holding the model + the port) survives and the next launch would reconnect to
# it instead of spawning fresh. Reap it so every launch runs the current code.
# (also the Ableton Link reader subprocess, which holds a network port).
STALE="$(pgrep -f "$ROOT/sidecar/server.py" || true) $(pgrep -f "$ROOT/sidecar/link_proc.py" || true)"
STALE="$(echo $STALE | xargs)"
if [ -n "$STALE" ]; then
  echo "Reaping orphaned sidecar(s): $STALE"
  kill $STALE 2>/dev/null || true
  sleep 1
  kill -9 $STALE 2>/dev/null || true
fi

APP="$ROOT/app/build/ACE15_artefacts/Release/Standalone/ACE15 Realtime.app"
echo "Launching $APP (sidecar auto-spawns; first run downloads ~6GB of models with a progress bar)"
exec "$APP/Contents/MacOS/ACE15 Realtime"
