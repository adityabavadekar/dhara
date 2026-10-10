"""Stage 0: pick the S1 before/after pairs on the same relative orbit for an AOI and event date."""

from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from pydantic import BaseModel
from shapely.geometry import shape
from shapely.ops import unary_union

from dhara.adapters.cdse import stac_search
from dhara.adapters.sentinel1 import S1Pass
from dhara.config import get_settings
from dhara.geo import AOI
from dhara.timeutil import event_day_utc


class SceneSelectionError(RuntimeError):
    pass


class S1Pair(BaseModel):
    """Same-relative-orbit pair. `pre2` (the pass before `pre`) is optional; the ML member uses it."""

    relative_orbit: int
    orbit_state: str
    pre: S1Pass
    post: S1Pass
    pre2: S1Pass | None = None

    @property
    def baseline_days(self) -> float:
        return (self.post.start - self.pre.start).total_seconds() / 86400


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def group_passes(items: list[dict[str, Any]], aoi: AOI) -> list[S1Pass]:
    """Group IW GRDH dual-pol (VV+VH) STAC items into passes and compute AOI coverage (EPSG:4326)."""
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        p = it["properties"]
        if p.get("sar:instrument_mode") != "IW" or not str(p.get("product:type", "")).startswith(
            "IW_GRDH"
        ):
            continue
        if not {"VV", "VH"} <= set(p.get("sar:polarizations", [])):
            continue
        dt = _parse_dt(p["datetime"])
        groups[(p["platform"], int(p["sat:relative_orbit"]), dt.strftime("%Y%m%d"))].append(it)

    aoi_geom = aoi.geometry
    passes: list[S1Pass] = []
    for (platform, rel, _), its in groups.items():
        # The same slice can appear more than once (e.g. reprocessed); dedupe by start time.
        uniq = {it["properties"]["datetime"]: it for it in its}
        its = sorted(uniq.values(), key=lambda i: i["properties"]["datetime"])
        foot = unary_union([shape(i["geometry"]) for i in its])
        cov = foot.intersection(aoi_geom).area / aoi_geom.area
        p0 = its[0]["properties"]
        passes.append(
            S1Pass(
                platform=platform,
                relative_orbit=rel,
                orbit_state=p0["sat:orbit_state"],
                start=_parse_dt(p0["datetime"]),
                end=max(
                    _parse_dt(i["properties"].get("end_datetime", i["properties"]["datetime"]))
                    for i in its
                ),
                scene_ids=[i["id"] for i in its],
                aoi_coverage=round(cov, 4),
            )
        )
    return sorted(passes, key=lambda p: p.start)


def select_pairs(
    passes: list[S1Pass],
    event: date | datetime,
    min_coverage: float = 0.5,
) -> tuple[list[S1Pair], list[str]]:
    """Per relative orbit: the last pass before the event, the one before it, and the first pass after.

    `event` is a date (UTC day, time unknown) or an aware datetime. With a date only, passes on
    the event day are ambiguous (before or after the event?) and are skipped, with a note.
    Returns pairs ranked by (coverage, time to post, baseline), plus human-readable notes.
    """
    notes: list[str] = []
    event_date = event_day_utc(event)
    if isinstance(event, datetime):
        t_event = event

        def is_pre(p: S1Pass) -> bool:
            return p.end < t_event

        def is_post(p: S1Pass) -> bool:
            return p.start > t_event
    else:

        def is_pre(p: S1Pass) -> bool:
            return p.start.date() < event_date

        def is_post(p: S1Pass) -> bool:
            return p.start.date() > event_date

        same_day = [p.key for p in passes if p.start.date() == event_date]
        if same_day:
            notes.append(
                f"Skipped passes on the event day (event time unknown): {', '.join(same_day)}"
            )

    by_orbit: dict[int, list[S1Pass]] = defaultdict(list)
    for p in passes:
        if p.aoi_coverage < min_coverage:
            notes.append(
                f"{p.key}: covers {p.aoi_coverage:.0%} of the AOI, below {min_coverage:.0%}"
            )
            continue
        by_orbit[p.relative_orbit].append(p)

    pairs: list[S1Pair] = []
    for rel, ps in sorted(by_orbit.items()):
        pre = [p for p in ps if is_pre(p)]
        post = [p for p in ps if is_post(p)]
        if not pre or not post:
            notes.append(
                f"Relative orbit {rel}: {len(pre)} pre-event and {len(post)} post-event passes; "
                "no pair"
            )
            continue
        pairs.append(
            S1Pair(
                relative_orbit=rel,
                orbit_state=pre[-1].orbit_state,
                pre=pre[-1],
                post=post[0],
                pre2=pre[-2] if len(pre) > 1 else None,
            )
        )

    def rank(pr: S1Pair) -> tuple[float, float, float]:
        cov = min(pr.pre.aoi_coverage, pr.post.aoi_coverage)
        lag = (pr.post.start.date() - event_date).days
        return (-round(cov, 1), lag, pr.baseline_days)

    pairs.sort(key=rank)
    return pairs, notes


def find_s1_pairs(aoi: AOI, event: date | datetime) -> tuple[list[S1Pair], list[str]]:
    """Query CDSE STAC and return ranked same-orbit pairs. Raises if none exist."""
    s = get_settings()
    day = event_day_utc(event)
    start = datetime.combine(day - timedelta(days=s.dhara_s1_search_days), time(0), tzinfo=UTC)
    end = datetime.combine(day + timedelta(days=s.dhara_s1_search_days + 1), time(0), tzinfo=UTC)
    items = stac_search("sentinel-1-grd", aoi.bounds, start, end)
    passes = group_passes(items, aoi)
    pairs, notes = select_pairs(passes, event, s.dhara_min_s1_coverage)
    if not pairs:
        raise SceneSelectionError(
            f"No Sentinel-1 same-orbit pre/post pair within ±{s.dhara_s1_search_days} days of "
            f"{day.isoformat()}. " + " | ".join(notes)
        )
    return pairs, notes
