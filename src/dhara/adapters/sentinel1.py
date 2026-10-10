"""Sentinel-1 GRD via CDSE: discovery (STAC) and AOI-only backscatter via the Sentinel Hub Process API."""

from datetime import timedelta
from typing import Any, Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel

from dhara.adapters.cdse import Grid, sh_process
from dhara.timeutil import iso_utc

Radiometry = Literal["GAMMA0_TERRAIN", "SIGMA0_ELLIPSOID"]


class S1Pass(BaseModel):
    """All GRD slices from one satellite pass (same platform, relative orbit, datatake) over the AOI."""

    platform: str
    relative_orbit: int
    orbit_state: str
    start: AwareDatetime  # UTC
    end: AwareDatetime  # UTC
    scene_ids: list[str]
    # Fraction of the AOI covered by the union of the slice footprints.
    aoi_coverage: float

    @property
    def key(self) -> str:
        return f"{self.platform}_R{self.relative_orbit:03d}_{self.start:%Y%m%dT%H%M%S}Z"


_EVAL_S1 = """//VERSION=3
function setup() {
  return {input: [{bands: [%s]}], output: {bands: %d, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  if (s.dataMask == 0) { return [%s]; }
  return [%s];
}
"""


def _evalscript(bands: list[str]) -> str:
    quoted = ", ".join(f'"{b}"' for b in [*bands, "dataMask"])
    nan = ", ".join(["NaN"] * len(bands))
    vals = ", ".join(f"s.{b}" for b in bands)
    return _EVAL_S1 % (quoted, len(bands), nan, vals)


def fetch_pass(
    p: S1Pass,
    grid: Grid,
    radiometry: Radiometry,
    *,
    with_geometry_bands: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Fetch linear backscatter for one pass on `grid` (grid CRS, usually local UTM).

    Server-side: orthorectified on COP-DEM GLO-30, LEE 5x5 speckle filter.
    `GAMMA0_TERRAIN` = radiometrically terrain-flattened γ⁰ (classical/debris members).
    `SIGMA0_ELLIPSOID` = σ⁰, matching Kuro Siwo preprocessing (ML member).
    Optional bands: localIncidenceAngle (deg) and shadowMask (non-zero = layover/shadow).
    Pixels outside the swath are NaN.
    """
    bands = ["VV", "VH"]
    if with_geometry_bands:
        bands += ["localIncidenceAngle", "shadowMask"]
    data: list[dict[str, Any]] = [
        {
            "type": "sentinel-1-grd",
            "dataFilter": {
                "timeRange": {
                    "from": iso_utc(p.start - timedelta(seconds=30)),
                    "to": iso_utc(p.end + timedelta(seconds=30)),
                },
                "acquisitionMode": "IW",
                "polarization": "DV",
                "resolution": "HIGH",
                "orbitDirection": p.orbit_state.upper(),
            },
            "processing": {
                "backCoeff": radiometry,
                "orthorectify": True,
                "demInstance": "COPERNICUS_30",
                "speckleFilter": {"type": "LEE", "windowSizeX": 5, "windowSizeY": 5},
            },
        }
    ]
    arr = sh_process(grid, data, _evalscript(bands), len(bands))
    return arr, bands
