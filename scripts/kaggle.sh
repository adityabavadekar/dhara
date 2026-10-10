#!/usr/bin/env bash
# Run commands / move files on the Kaggle GPU kernel (training only).
#   scripts/kaggle.sh run "nvidia-smi"
#   scripts/kaggle.sh put training/ /kaggle/working/
#   scripts/kaggle.sh get /kaggle/working/runs/x.pt models/
# Reads KAGGLE_RELAY_URL, KAGGLE_RELAY_SECRET, KAGGLE_KERNEL_ID from .env.
# Wraps project-clones/kaggle-terminal/kssh.sh (clone: github.com/adityabavadekar/kaggle-terminal).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; [[ -f "$ROOT/.env" ]] && source "$ROOT/.env"; set +a
: "${KAGGLE_RELAY_URL:?set KAGGLE_RELAY_URL in .env}" "${KAGGLE_RELAY_SECRET:?set KAGGLE_RELAY_SECRET in .env}"
KSSH="$ROOT/project-clones/kaggle-terminal/kssh.sh"
[[ -x "$KSSH" ]] || { echo "missing $KSSH; git clone https://github.com/adityabavadekar/kaggle-terminal project-clones/kaggle-terminal" >&2; exit 1; }
RELAY_URL="$KAGGLE_RELAY_URL" RELAY_SECRET="$KAGGLE_RELAY_SECRET" KERNEL_ID="${KAGGLE_KERNEL_ID:-}" exec bash "$KSSH" "$@"
