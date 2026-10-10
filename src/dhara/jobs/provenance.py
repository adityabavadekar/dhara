"""Provenance sidecar written next to every job's outputs."""

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import dhara
from dhara.timeutil import now_utc


def git_state() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=5
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    return {
        "commit": _run("rev-parse", "HEAD") or "uncommitted",
        "dirty": bool(_run("status", "--porcelain")),
    }


class Provenance:
    def __init__(self, job_id: str, inputs: dict[str, Any]):
        self.doc: dict[str, Any] = {
            "job_id": job_id,
            "created_utc": now_utc(),
            "dhara_version": dhara.__version__,
            "git": git_state(),
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "rasterio", "geopandas", "shapely")},
            "inputs": inputs,
            "data_sources": {},
            "parameters": {},
            "stages": {},
            "notes": [],
        }

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        entry: dict[str, Any] = {"status": "running"}
        self.doc["stages"][name] = entry
        try:
            yield entry
            entry["status"] = "ok"
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - t0, 2)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.doc, indent=2, default=str))
