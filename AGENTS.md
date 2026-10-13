# AGENTS.md

Dhara maps flood and debris damage, damaged OSM assets and cut-off settlements for any AOI and date.
It uses only Sentinel-1/2, Copernicus DEM, pre-event OSM, and Kuro Siwo/Sen1Floods11 for training. See `PLAN.md`.

## Hard rules (disqualification risk)
- All network access goes through `src/dhara/adapters/` → `allowlist.guarded_request()`. Never call httpx/requests directly anywhere else.
- `src/dhara/` must never read, import or mention `validation/` or any reference-map source (CEMS/EMSR, UNOSAT, GFM, OPERA DSWx/DIST, HOT/fAIr, WorldPop, WorldCover, JRC, MERIT, Overpass). `tests/test_compliance.py` enforces this.
- ohsome timestamps must be ≤ `2026-07-27`.
- Foundation-model weights stay behind `allow_foundation_weights` (default false).

## Engineering rules
- Python 3.12, `uv`. Lint and type-check with `uv run ruff check . && uv run ruff format --check . && uv run ty check`. Run tests with `uv run pytest`.
- Every stage is a module with typed (pydantic) input/output contracts.
- Every function that handles geodata states its CRS in the docstring. Analysis runs in local UTM; the API uses EPSG:4326.
- Timestamps are ISO 8601 and always timezone-aware; store and output them as UTC `…Z`. Use `dhara.timeutil` and never create naive datetimes.
- Frontend: shadcn/ui, following the components and layout of `project-clones/shadcn-admin`. Flat and professional, with no glow, gradients or neon.
- GPU training runs on Kaggle through `scripts/kaggle.sh` (run/put/get). Pipeline code never calls it.
- Every job writes a provenance sidecar (scene IDs, git commit, parameters, timings).
- Cache key: (AOI hash, date, stage).
- Fail loudly with actionable messages. "Not observable" is never shown as "no damage".
- Never invent numbers in code, docs or UI copy. Write TODO until a value is computed.
- Never show images of victims. The UI and reports state that this is an educational prototype.
- Abstract complexity from the UI: This application is for disaster researchers and emergency decision-makers. Never expose internal implementation details, algorithm mechanics (e.g. Dijkstra, raw graph internals), or backend AI provider/model IDs (e.g. "groq", "openai/gpt-oss-120b", "gemini") in user-facing UI copy or badges. Keep all UI terminology clean, professional, and domain-focused (e.g., "AI Assistant", "Grounded Evidence").
- `project-clones/` is read-only reference material. Copy code from it; don't import it.
- Don't commit unless the user asks.
