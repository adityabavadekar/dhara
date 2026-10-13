# Check against the HDX / Microsoft AI for Good reference (validation only)

Reference: `nepal_flash_flood_2026_overture_buildings_damage.gpkg` (HDX, Microsoft AI for Good
Lab): Overture building footprints overlaid on the UNOSAT damage area (product 4256). 4,977 of
633,208 buildings are marked `damaged = 1`. Used only to check results; `src/dhara` never reads it.

Method (`validation/compare_buildings.py`), inside each job's AOI:
- **recall** = share of reference-damaged buildings within 50 m of a mapped flood or debris pixel;
- **precision** = share of our "hit" buildings within 50 m of a reference-damaged building.
OSM and Overture footprints differ, so matching is by distance, not id.

| Job | Event date | Radar pair | AOI km² | Ref. damaged | Our hits | Recall | Precision |
|---|---|---|---:|---:|---:|---:|---:|
| c8fb520f6174 | 28 Aug | 24 Aug → 5 Sep | 1,585 | 1,331 | 279 | 0.85 | 0.94 |
| 5c20bca1a959 | 28 Aug | 24 Aug → 5 Sep | 192 | 384 | 79 | 0.84 | 0.86 |
| 68d2b5869583 | 26 Aug | 16 Aug → 28 Aug | 818 | 3,332 | 2,901 | 0.99 | 0.70 |
| 6f18034fa3c4 | 28 Aug | 16 Aug → 28 Aug | 27 | 654 | 254 | 0.99 | 1.00 |
| befc63b2f169 | 20 Aug | **12 Aug → 24 Aug (both before the flood)** | 27 | 654 | 220 | 0.92 | 1.00 |
| bd970069311d | 12 Aug | **4 Aug → 16 Aug (both before the flood)** | 731 | 382 | 873 | 0.97 | **0.16** |

How to read this:
- The wide pre-event control (bd970069311d) is the meaningful false-alarm test: on two images taken
  before the flood, 84% of our "hits" are not near reference damage. Ordinary radar change in the
  valleys is mapped as debris; this is the largest weakness.
- befc63b2f169 also uses only pre-event images yet scores precision 1.00: in a 27 km² strip along
  the river, almost every riverside building is within 50 m of a reference-damaged one, so this
  precision measure says little for narrow AOIs. Recall near 1.0 in narrow AOIs is inflated the
  same way.
- A 50 m tolerance is generous at a 10 m grid. These are upper bounds, not accuracy figures.
