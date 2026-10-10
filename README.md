# Dhara

Give Dhara an area and a flood date. It maps where the flood hit, which pre-event OSM buildings and roads were affected, and (coming in a later milestone) which settlements are cut off. It uses only Sentinel-1/2, Copernicus DEM GLO-30 and OpenStreetMap from before the event.

Status: milestone 1. See [PLAN.md](PLAN.md).

## Setup
```bash
uv sync
cp .env.example .env   # fill in secrets
```

## Run
```bash
uv run dhara scenes --bbox 85.25,27.95,85.40,28.20 --event 2026-08-26
uv run dhara run --bbox 85.30,28.15,85.33,28.175 --event 2026-08-26
```

Outputs go to `data/jobs/<job_id>/`:
- `classes.tif`: 0 not observable, 1 no change, 2 permanent water, 3 flood
- `p_flood_classical.tif`
- `buildings.parquet` and `roads.parquet`
- `*_hit.geojson`
- `summary.json`
- `provenance.json`

## Tests
```bash
uv run pytest                          # offline, includes compliance guardrails
DHARA_LIVE=1 uv run pytest -m live     # real CDSE + ohsome
```

## Attributions
- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors (ODbL), via the ohsome API (HeiGIT).
- Kuro Siwo: Bountos et al., NeurIPS 2024. Sen1Floods11: Bonafilia et al., CVPR Workshops 2020.
