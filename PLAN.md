# Dhara: PLAN

## 1. Task and judging
- Input: an AOI (bbox or polygon) and a flood date. Output: a flood/debris map, the OSM assets that were hit, the settlements that are cut off, a map dashboard and a one-page sitrep.
- Inputs allowed: S1, S2, COP-DEM GLO-30, OSM via ohsome at or before 2026-07-27, Kuro Siwo, Sen1Floods11. Using anything else means disqualification.
- Judges run it live on AOIs and dates they choose, so a full run on a new AOI has to finish in minutes. Nothing can be specific to Rasuwa.
- Weights: mapping 30, AI 25, cut-off 20, report/Q&A/honesty 15, dashboard/sitrep 10.
- Case study: EMSR927, used only for validation. The CDSE STAC has these S1D same-track pairs:
  T85 asc 08-16 → 08-28, T121 desc 08-19 → 08-31, T19 desc 08-24 → 09-05.
  Only S1D appears over this AOI, so each track repeats every 12 days.

## 2. Decisions
| Area | Decision | Reason |
|---|---|---|
| AI component | **(a)** UNet-ResNet50 on Kuro Siwo pre/pre/post GRD. Generalization is tested by holding out high-relief events and running zero-shot on Rasuwa. The sitrep stays deterministic. | (a) feeds the mapping and cut-off scores (75% of the total). (b) only scores inside the 25%. |
| SAR route | Find scenes in the **CDSE STAC**, which is public and needs no auth. Fetch pixels with the **CDSE Sentinel Hub Process API**: AOI only, orthorectified on COPDEM30, LEE filter. | No full-GRD download and no SNAP in the live path. |
| SAR radiometry | The ML member gets **σ⁰ ellipsoid, linear**. Classical and debris members get **γ⁰ terrain-flattened**. The layover/shadow mask comes from SH `shadowMask` plus local incidence angle. | Kuro Siwo's SNAP graph is σ⁰ + Lee Sigma + Terrain-Correction with no flattening, so the model input has to match it. |
| DEM | COP-DEM GLO-30 through the same SH API (`demInstance=COPERNICUS_30`). | One adapter and one allowlist. |
| Terrain | pysheds (D8, accumulation, HAND, conditioning). | Pure Python. WhiteboxTools downloads a binary at runtime. |
| Network | networkx on a graph we build from ohsome geometries. No osmnx. | osmnx calls Overpass, which serves current OSM and is not allowed. |
| Storage | COG + GeoParquet per job, with **DuckDB (spatial)** as the catalog, facts table and job table. | One file, no server, and the sitrep can query GeoParquet directly. PostGIS adds ops work and we don't have concurrent writers. |
| Queue | In-process worker (FastAPI + ProcessPool, max 2 jobs) with job state in DuckDB. | Enough for a demo. Redis can be added later if needed. |
| Python | 3.12, pinned with `uv` (the machine has 3.14). | Wheel support for torch, rasterio and pysheds. |
| Frontend | React + Vite + MapLibre + **shadcn/ui**. Layout and components follow `satnaing/shadcn-admin` (sidebar, data-table, TanStack Router/Query). Flat, no glow. Type-checked with `tsgo`. | Your call: a professional, premium feel. |
| Training compute | Kaggle 2x T4, reached over SSH with `scripts/kaggle.sh` (a kaggle-terminal relay). Kuro Siwo shards stream from HuggingFace. Only checkpoints and metrics persist in `/kaggle/working`. | Labelled GRD is about 170 GB (train 46.8 + test 122.8). `/kaggle/working` holds 20 GB; the root disk is ephemeral with 1.1 TB free. |
| Timestamps | ISO 8601, always timezone-aware. Input is a date or a datetime with an offset; naive datetimes are rejected. Stored and output as UTC `…Z`. | `dhara/timeutil.py`. A local event time (+05:45) correctly resolves same-day passes. |
| Gray areas | We decide ourselves, without asking organizers. FM weights and test-time self-training stay **OFF** by default, and the report discloses both flags. | Your call. |
| OSM route | **ohsome v2** (`api.heigit.org`, free key) as the main route, returning GeoParquet. The fallback is v1 `elementsFullHistory` filtered to the snapshot. | v1 `elements/geometry` now returns 403 and v1 shuts down on 2026-11-30. The fallback works but is slow: 68 s for 782 buildings. |
| Package name | `src/dhara/` (the brief says "cutline/" in one place, which I read as the old name). | |

## 3. Open questions
1. **Hosted instance.** Where will it run (VM size), and which CDSE account does it use? SH Process units are metered per account.
2. **OSM timestamp.** I plan to use `min(UTC event day − 1 day, 2026-07-27)` at 00:00Z for judge dates. OK?
3. **AOI size cap for live runs.** I propose ≤ 2,500 km², with a clear error above that.
4. **Server-side ortho.** Does SH-processed S1 count as "raw satellite data"? I'm assuming yes, because it is CDSE and the same as GRD processing.
5. **Nepali sitrep.** Only a bonus under (a). Should I build it with templates?
6. **Credentials needed in `.env`.** A CDSE Sentinel Hub OAuth client is required to finish milestone 1 live. An ohsome v2 API key is strongly recommended.

## 4. Milestones
1. **Vertical slice (CLI).** AOI + date → scene selection → S1 fetch → log-ratio/Otsu member → ohsome buildings/roads → hit overlay → GeoJSON + provenance sidecar. Also: the allowlist, the compliance test and a smoke test.
   *Status:* code done and 90 offline tests pass. Scene selection and ohsome are verified live. The S1 fetch has not run live because CDSE credentials are missing.
2. Terrain (HAND, D8) and fusion, with the "not observable" class.
3. ML member: Kuro Siwo training, held-out mountain evaluation, inference adapter.
4. Damage overlay with confidence tiers and the bridge check.
5. Cut-off analysis: reachability, critical edges, bridge-repair ranking, Monte Carlo P(cut-off).
6. Sitrep (HTML/PDF, numeral verifier).
7. API and dashboard, plus Docker Compose.
8. Validation against EMSR927 (`validation/`).
9. Report and video assets.
10. Bonus: flow-path tracing.

## 5. Risks and mitigations
| Risk | Mitigation |
|---|---|
| CDSE quota or rate limits (SH Process units, 429s) | Fetch the AOI only, cache by (AOI hash, date, stage), retry with backoff, and enforce the AOI size cap. |
| No same-track S1 pair | Search ±24 days on every relative orbit and fail loudly with the reason. Fall back to S2-only (marked low confidence). |
| Layover and shadow in gorges | Fuse ascending and descending tracks. Masked pixels are classed "not observable", never "no damage". |
| Model domain shift (lowlands → Himalaya) | Match σ⁰ preprocessing, hold out high-relief events, use the HAND prior, and keep the classical and debris members in the ensemble. Report the drop honestly. |
| Debris is not water | A separate bidirectional-change member plus S2 NDVI-loss/BSI. |
| Live-run latency | No full scenes, cached terrain, CPU inference on tiles, and per-stage timings in provenance. |
| Incomplete OSM (missing trails or bridges) | Report graph coverage in the sitrep and state it as a limitation. |

## 6. Compliance checklist and enforcement
| Rule | Enforcement in code |
|---|---|
| Only allowed hosts and collections | `adapters/allowlist.py`: every HTTP call goes through `guarded_request()`, which checks the host and collection. Anything else raises `ComplianceError`. |
| OSM ≤ 2026-07-27 | The ohsome adapter clamps the timestamp and raises if a later one is requested. The test also scans for literal timestamps. |
| No reference maps in the pipeline | `tests/test_compliance.py` fails if anything under `src/dhara/` mentions `validation`, `emergency.copernicus`, `unosat`, `hotosm`, `gfm`, `opera`, `worldpop`, `worldcover`, `jrc`, `merit`, or `overpass`. |
| Training only on Kuro Siwo or Sen1Floods11 | The training config whitelists dataset names, and the model card records lineage. |
| FM weights are a gray area | `models.allow_foundation_weights: false` by default, and the code refuses to load FM weights unless it is set. |
| Provenance | Every job writes a `provenance.json`: scene IDs, ohsome timestamp, git commit, model version, parameters, timings. |
