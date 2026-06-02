#!/bin/bash
# Dev launcher: runs the standalone app, which auto-spawns the Python sidecar.
# (In the shipped .app the sidecar + a private python live inside the bundle.)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ACE15_SPAWN_SIDECAR=1
export ACE15_PYTHON="$ROOT/.venv/bin/python"
export ACE15_SIDECAR="$ROOT/sidecar/server.py"
export PYTORCH_ENABLE_MPS_FALLBACK=1
APP="$ROOT/app/build/ACE15_artefacts/Release/Standalone/ACE15 Realtime.app"
echo "Launching $APP (sidecar auto-spawns; first run downloads ~6GB of models with a progress bar)"
exec "$APP/Contents/MacOS/ACE15 Realtime"
