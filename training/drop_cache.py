"""Every `period` s, ask the kernel to drop cached pages of the training data and run files.
The files stay on disk; only their page-cache copies are released. Kaggle counts page cache in the
container's memory (cgroup memory.current), so a full cache can look like memory exhaustion.
Usage on the GPU box: python drop_cache.py /tmp/ks_full /kaggle/working/dhara/runs &"""

import os
import sys
import time


def drop(root: str) -> None:
    for d, _, files in os.walk(root):
        for f in files:
            try:
                fd = os.open(os.path.join(d, f), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


if __name__ == "__main__":
    roots = sys.argv[1:] or ["/tmp/ks_full"]
    while True:
        for r in roots:
            drop(r)
        time.sleep(60)
