#!/usr/bin/env bash
# Run MuScriptor from the .venv made by ./install.sh, e.g.
#   ./run.sh serve [--model large]
#   ./run.sh transcribe song.mp3
# Calls the venv directly (no `uv run`): no lockfile check on every start, and
# a PyTorch build swapped in by install.sh for an older driver stays in place.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$here/.venv/bin/muscriptor" ]]; then
    echo "No .venv yet: run ./install.sh first" >&2
    exit 1
fi
exec "$here/.venv/bin/muscriptor" "$@"
