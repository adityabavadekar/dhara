#!/usr/bin/env bash
# Resource log every 15 s (memory incl. cgroup counters, GPU, disk, top processes) for diagnosing
# session restarts during training. Usage: bash sysmon.sh /kaggle/working/sysmon.log &
OUT="${1:-/kaggle/working/sysmon.log}"
cg() { for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/memory/memory.limit_in_bytes; do [ -r "$f" ] && echo "$(basename "$f")=$(( $(cat "$f" 2>/dev/null || echo 0) / 1048576 ))MiB"; done 2>/dev/null | tr '\n' ' '; }
while true; do
  {
    echo "== $(date -u +%FT%TZ) up=$(cut -d' ' -f1 /proc/uptime)s"
    free -m | sed -n 2,3p
    echo "cgroup: $(cg)"
    grep -E "^(Shmem|Dirty|Cached|MemAvailable):" /proc/meminfo | tr -s ' ' | tr '\n' ' '; echo
    df -m /dev/shm /tmp /kaggle/working | tail -n 3
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    ps -eo rss,pid,etime,args --sort=-rss | head -n 6 | cut -c1-140
  } >> "$OUT"
  sleep 15
done
