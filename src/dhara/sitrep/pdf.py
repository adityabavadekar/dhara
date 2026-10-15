"""Server-side PDF of the one-page situation report: headless Chromium prints the same React page
the browser shows (/labs/reports), served by this API from the built frontend (web/dist).

The browser may only reach the local API: every other host is unresolvable for it, so rendering a
report never touches a third-party service. No geodata is handled here.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
RENDER_TIMEOUT_S = 180


class ReportRenderError(RuntimeError):
    pass


def report_filename(facts: dict[str, str], job_id: str, lang: str) -> str:
    """Dhara_SitRep_<event day>_<place>_<job id>[_NE], without extension. Same rule as
    `reportFileName` in web/src/features/labs/report-studio.tsx."""
    day = (facts.get("event") or "")[:10]
    place = (facts.get("target_location") or "").split(" (")[0]
    # Keep letters, combining marks (Devanagari vowel signs) and digits; anything else becomes "-".
    place = "".join(c if unicodedata.category(c)[0] in "LMN" else "-" for c in place)
    place = re.sub("-+", "-", place).strip("-")
    lat, lon = facts.get("target_lat"), facts.get("target_lon")
    where = place or (f"{lat}N-{lon}E" if lat and lon else "")
    name = "_".join(p for p in ("Dhara_SitRep", day, where, job_id) if p)
    return name + ("_NE" if lang == "ne" else "")


def ascii_filename(facts: dict[str, str], job_id: str, lang: str) -> str:
    """The same name with a non-ASCII place replaced by its coordinates, for old clients."""
    name = report_filename(facts, job_id, lang)
    if name.isascii():
        return name
    return report_filename({**facts, "target_location": ""}, job_id, lang)


def find_browser() -> str:
    for b in BROWSERS:
        if path := shutil.which(b):
            return path
    raise ReportRenderError(
        "No Chromium/Chrome found for the PDF report. Install chromium (e.g. `apt install chromium`) "
        "or use the Print button on the report page."
    )


def render_pdf(page_url: str, out: Path, port: int) -> None:
    """Print `page_url` (the report page on this API, http://127.0.0.1:<port>/...) to `out`."""
    browser = find_browser()
    with tempfile.TemporaryDirectory(prefix="dhara-pdf-") as profile:
        tmp = out.with_suffix(".tmp.pdf")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            # Only the local API resolves; fonts are bundled with the frontend.
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            # Virtual time stops while requests are pending, so the report data and images load first.
            "--virtual-time-budget=30000",
            f"--print-to-pdf={tmp}",
            page_url,
        ]
        if hasattr(os, "geteuid") and os.geteuid() == 0:  # Chromium refuses to sandbox as root
            cmd.insert(1, "--no-sandbox")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=RENDER_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise ReportRenderError(
                f"Rendering the report took longer than {RENDER_TIMEOUT_S} s"
            ) from e
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            raise ReportRenderError(
                f"Chromium could not print the report (exit {r.returncode}): {r.stderr[-400:]}"
            )
        tmp.replace(out)
