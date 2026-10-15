#!/usr/bin/env bash
# One-step setup for Dhara: Python environment, dashboard, .env and model weights.
#   scripts/setup.sh                 everything
#   scripts/setup.sh --weights-only  only download and verify the model weights
#   scripts/setup.sh --no-web        skip the dashboard (command line only)
# Safe to run again: finished steps are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."

WEIGHTS_URL="https://huggingface.co/adityabavadekar/dhara-flood-unet/resolve/main/model.onnx"
WEIGHTS_SHA256="038c6d80eab8a40639f2c64c5ab79036548b7eb5719d913dd3fd8e1d8a13005a"
WEIGHTS_PATH="models/flood_unet/model.onnx"

WEB=1
WEIGHTS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --no-web) WEB=0 ;;
    --weights-only) WEIGHTS_ONLY=1 ;;
    -h | --help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
fail() { printf 'Error: %s\n' "$1" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is not installed. $2"; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

download_weights() {
  step "Model weights ($WEIGHTS_PATH, 98 MB)"
  if [[ -f "$WEIGHTS_PATH" && "$(sha256 "$WEIGHTS_PATH")" == "$WEIGHTS_SHA256" ]]; then
    echo "Already present and verified."
    return
  fi
  need curl "Install curl, or download $WEIGHTS_URL to $WEIGHTS_PATH by hand."
  local tmp="$WEIGHTS_PATH.part"
  curl -fL --retry 3 --progress-bar -o "$tmp" "$WEIGHTS_URL" \
    || fail "Download failed. Check your connection, or download $WEIGHTS_URL to $WEIGHTS_PATH by hand."
  local got
  got="$(sha256 "$tmp")"
  if [[ "$got" != "$WEIGHTS_SHA256" ]]; then
    rm -f "$tmp"
    fail "Checksum mismatch (got $got). The download is incomplete or changed; run this script again."
  fi
  mv "$tmp" "$WEIGHTS_PATH"
  echo "Downloaded and verified."
}

if [[ "$WEIGHTS_ONLY" == 1 ]]; then
  download_weights
  exit 0
fi

step "Checking prerequisites"
need uv "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
if [[ "$WEB" == 1 ]]; then
  need node "Install Node.js 20 or newer (https://nodejs.org), or run with --no-web."
  need pnpm "Install it with: npm install -g pnpm (or corepack enable), or run with --no-web."
  node_major="$(node -p 'process.versions.node.split(".")[0]')"
  [[ "$node_major" -ge 20 ]] || fail "Node.js $node_major found; Dhara needs 20 or newer."
fi
echo "OK"

step "Python environment (uv sync)"
uv sync

if [[ "$WEB" == 1 ]]; then
  step "Dashboard dependencies (pnpm install)"
  (cd web && pnpm install)
fi

step "Configuration (.env)"
if [[ -f .env ]]; then
  echo ".env already exists; left unchanged."
else
  cp .env.example .env
  echo "Created .env from .env.example."
fi

download_weights

step "Checking the installation"
uv run dhara --help >/dev/null && echo "dhara command works."

cat <<'EOF'

Setup complete.

Next:
  1. To analyse a new area or date, add your free Copernicus Data Space credentials to .env
     (CDSE_SENTINEL_HUB_CLIENT_ID, CDSE_SENTINEL_HUB_CLIENT_SECRET). See README, Installation.
  2. Start the API:        uv run dhara serve
  3. Start the dashboard:  cd web && pnpm dev     (then open http://localhost:5173)
EOF
