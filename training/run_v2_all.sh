#!/usr/bin/env bash
# Prep all Kuro Siwo shards, then smoke test + full training. Detached-friendly; logs to /kaggle/working.
set -euo pipefail
cd "$(dirname "$0")"
setsid bash sysmon.sh /kaggle/working/sysmon.log < /dev/null > /dev/null 2>&1 &
bash prep_all.sh /tmp/ks_full
bash train_v2.sh
echo "ALL DONE"
