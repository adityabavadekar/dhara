"""Compare Dhara's bridge assessment with the damaged bridges in Copernicus EMS EMSR927.
Validation only: nothing here is read by `src/dhara`.
Credit: "European Union, Copernicus Emergency Management Service data".

EMS marks damaged bridges as points (Destroyed, Damaged or Possibly damaged); it does not list
bridges it saw standing, so only recall can be measured. For each EMS area that overlaps a job,
every EMS bridge inside the job area is matched to the nearest OpenStreetMap bridge Dhara assessed
(within MATCH_M, in the job's UTM CRS) and counted as:

- flagged: Dhara marked that bridge likely lost or at risk;
- missed: Dhara assessed it as likely standing;
- not mapped: no OpenStreetMap bridge within MATCH_M, so Dhara could not assess it.

It also lists Dhara's likely-lost bridges in the same region that EMS did not mark (EMS may have
missed them, or Dhara is wrong; there is no ground truth for those).

Usage: uv run python validation/compare_bridges.py <job_id> [...]   (DHARA_JOBS = other job folder)
"""

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_emsr927 import ROOT, _layer, ems_aois  # noqa: E402

MATCH_M = 50.0
DAMAGED = ("Destroyed", "Damaged", "Possibly damaged")
FLAGGED = ("likely_lost", "at_risk")


def compare(job_id: str, jobs: Path | None = None) -> list[dict]:
    d = (jobs or Path(os.environ.get("DHARA_JOBS", ROOT / "data" / "jobs"))) / job_id
    if not (d / "bridges.parquet").exists():
        return []
    prov = json.loads((d / "provenance.json").read_text())
    ours = gpd.read_parquet(d / "bridges.parquet")
    crs = ours.crs
    job_aoi = gpd.GeoSeries([shape(prov["inputs"]["aoi"]["geojson"])], crs=4326).to_crs(crs)
    out = []
    for n, (name, ext) in ems_aois().items():
        ems = _layer(n, "transportationP")
        if ems is None or not len(ems):
            continue
        region = gpd.GeoSeries([ext], crs=4326).to_crs(crs).intersection(job_aoi.iloc[0]).iloc[0]
        if region.is_empty:
            continue
        ems = ems[ems["damage_gra"].isin(DAMAGED)].to_crs(crs)
        ems = ems[ems.within(region)]
        if not len(ems):
            continue
        cand = ours[ours.intersects(region.buffer(MATCH_M))]
        counts = {"flagged": 0, "missed": 0, "not_mapped": 0}
        matched: set = set()
        for pt in ems.geometry:
            dist = cand.distance(pt)
            if not len(dist) or dist.min() > MATCH_M:
                counts["not_mapped"] += 1
                continue
            i = dist.idxmin()
            matched.add(i)
            counts["flagged" if cand.loc[i, "status"] in FLAGGED else "missed"] += 1
        assessed = counts["flagged"] + counts["missed"]
        ours_lost = cand[(cand["status"] == "likely_lost") & cand.within(region)]
        out.append(
            {
                "job": job_id,
                "ems_aoi": f"{n:02d} {name}",
                "ems_damaged_bridges": len(ems),
                **counts,
                "recall_of_assessed": round(counts["flagged"] / assessed, 3) if assessed else None,
                "recall_of_all": round(counts["flagged"] / len(ems), 3),
                "ours_likely_lost": len(ours_lost),
                "ours_likely_lost_not_in_ems": int((~ours_lost.index.isin(matched)).sum()),
            }
        )
    return out


if __name__ == "__main__":
    for jid in sys.argv[1:]:
        for row in compare(jid):
            print(json.dumps(row))
