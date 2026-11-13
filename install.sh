#!/usr/bin/env bash
# One-shot installer for Linux + NVIDIA GPU (tuned for Ubuntu with an RTX 30xx).
#
#   ./install.sh                  # everything: uv, .venv, web UI, HF login
#   ./install.sh --model large    # also pre-download the large weights
#   ./install.sh --no-apt         # don't touch system packages (no sudo)
#
# Safe to re-run: every step skips what is already in place.
# Afterwards: ./run.sh serve   or   ./run.sh transcribe song.mp3
set -euo pipefail

cd "$(dirname "$0")"

MODEL=""
APT=1
WEB=1
HF=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --no-apt) APT=0; shift ;;
        --no-web) WEB=0; shift ;;
        --no-hf) HF=0; shift ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

bold() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# torch on PyPI is built against CUDA 13.0, which needs driver 580+.
MIN_DRIVER=580

# ---------------------------------------------------------------- GPU / driver
bold "GPU"
DRIVER_MAJOR=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRIVER_MAJOR=${DRIVER%%.*}
    echo "$GPU_NAME, driver $DRIVER"
    if (( DRIVER_MAJOR < MIN_DRIVER )); then
        warn "Driver $DRIVER is older than $MIN_DRIVER: a matching older CUDA build of"
        warn "PyTorch will be installed instead. Updating is the better fix:"
        warn "  sudo ubuntu-drivers install   (or: sudo apt install nvidia-driver-$MIN_DRIVER), then reboot"
    fi
else
    warn "nvidia-smi not found or not working: no NVIDIA driver, MuScriptor would run on CPU."
    warn "Install one with: sudo ubuntu-drivers install, reboot, then re-run ./install.sh"
fi

# ---------------------------------------------------------------- system packages
# fluidsynth: audio rendering of the MIDI (/auralize in the web UI).
if (( APT )) && command -v apt-get >/dev/null 2>&1; then
    if ! command -v fluidsynth >/dev/null 2>&1; then
        bold "System packages (sudo apt-get install fluidsynth)"
        sudo apt-get update -qq && sudo apt-get install -y -qq fluidsynth \
            || warn "Could not install fluidsynth; everything except MIDI playback rendering works."
    fi
fi

# ---------------------------------------------------------------- uv
bold "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

# ---------------------------------------------------------------- venv + deps
bold "Python environment (.venv)"
# Exact versions from uv.lock; uv fetches its own Python 3.13 if needed.
# Bytecode compiled now so the first start isn't slowed down by it.
uv sync --frozen --no-dev --compile-bytecode

PY=.venv/bin/python
cuda_ok() { "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if (( DRIVER_MAJOR > 0 )) && ! cuda_ok; then
    # Typically an older driver: let uv pick the PyTorch CUDA build it supports.
    bold "Reinstalling PyTorch for driver $DRIVER"
    TORCH_VER=$("$PY" -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])")
    TA_VER=$("$PY" -c "import importlib.metadata as m; print(m.version('torchaudio').split('+')[0])")
    uv pip install --python "$PY" --torch-backend=auto --reinstall-package torch \
        --reinstall-package torchaudio "torch==$TORCH_VER" "torchaudio==$TA_VER" \
        || warn "No PyTorch $TORCH_VER build for this driver: update the driver (see above)."
    warn "Use ./run.sh (not 'uv run', which would restore the CUDA 13 build from uv.lock)."
fi

"$PY" - <<'EOF'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"PyTorch {torch.__version__}: {p.name}, {p.total_memory / 2**30:.0f} GB, "
          f"sm_{p.major}{p.minor}")
else:
    print(f"PyTorch {torch.__version__}: CUDA NOT available, running on CPU")
EOF

# ---------------------------------------------------------------- web UI
if (( WEB )) && [[ ! -f muscriptor/web_dist/index.html ]]; then
    bold "Web UI"
    # Build from source with pnpm when it is there (needs a recent Node).
    # Otherwise, or if that fails, take the prebuilt UI out of the published
    # wheel of the same version (it ships there, see pyproject's artifacts).
    if ! { command -v pnpm >/dev/null 2>&1 \
            && (cd web && pnpm install --frozen-lockfile && pnpm run build); }; then
        echo "Fetching the prebuilt web UI from PyPI"
        "$PY" -I - <<'EOF'
import io, json, pathlib, tomllib, urllib.request, zipfile
version = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"]
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/muscriptor/{version}/json"))
url = next(f["url"] for f in meta["urls"] if f["filename"].endswith(".whl"))
with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())) as whl:
    names = [n for n in whl.namelist() if n.startswith("muscriptor/web_dist/")]
    whl.extractall(".", members=names)
print(f"web UI {version}: {len(names)} files")
EOF
    fi
fi

# ---------------------------------------------------------------- HuggingFace
if (( HF )); then
    bold "HuggingFace"
    if [[ -n "${HF_TOKEN:-}" ]] || .venv/bin/hf auth whoami >/dev/null 2>&1; then
        echo "Logged in."
    elif [[ -t 0 ]]; then
        echo "The weights need a HuggingFace login and the accepted license (CC BY-NC 4.0):"
        echo "  https://huggingface.co/MuScriptor/muscriptor-medium  (and -small / -large)"
        echo "Token: https://huggingface.co/settings/tokens (read access is enough)"
        .venv/bin/hf auth login || warn "Not logged in; run: .venv/bin/hf auth login"
    else
        warn "Not logged in to HuggingFace; run: .venv/bin/hf auth login"
    fi
fi

if [[ -n "$MODEL" ]]; then
    bold "Downloading the $MODEL weights"
    "$PY" - "$MODEL" <<'EOF'
import sys
from muscriptor.transcription_model import _resolve_config, _resolve_source
from muscriptor.utils.download import download_if_necessary
source = _resolve_source(sys.argv[1])
_resolve_config(source, download_if_necessary(source))
EOF
fi

bold "Done"
cat <<'EOF'
  ./run.sh serve                        web UI on http://127.0.0.1:8222
  ./run.sh serve --model large          most accurate model (fits the 3090 easily)
  ./run.sh transcribe song.mp3          MIDI next to the file
  ./run.sh transcribe --help            all options
EOF
