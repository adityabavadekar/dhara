from datetime import UTC, date, datetime

from dhara.geo import AOI
from dhara.jobs.scene_selection import group_passes, select_pairs

AOI_ = AOI.from_bbox(85.25, 27.95, 85.40, 28.20)
FULL = {
    "type": "Polygon",
    "coordinates": [[[84.0, 27.0], [86.0, 27.0], [86.0, 29.0], [84.0, 29.0], [84.0, 27.0]]],
}


def _item(dt: str, rel: int, state: str = "ascending", geom=FULL, pols=("VV", "VH")):
    return {
        "id": f"S1D_{dt}_{rel}",
        "geometry": geom,
        "properties": {
            "datetime": dt,
            "end_datetime": dt,
            "platform": "sentinel-1d",
            "sat:relative_orbit": rel,
            "sat:orbit_state": state,
            "sar:instrument_mode": "IW",
            "product:type": "IW_GRDH_1S",
            "sar:polarizations": list(pols),
        },
    }


ITEMS = [
    _item("2026-08-04T12:21:40Z", 85),
    _item("2026-08-16T12:21:41Z", 85),
    _item("2026-08-28T12:21:41Z", 85),
    _item("2026-08-26T00:18:44Z", 19, "descending"),  # same day as event
    _item("2026-09-07T00:18:44Z", 19, "descending"),
    _item("2026-08-14T00:18:44Z", 19, "descending"),
    _item("2026-08-20T00:10:00Z", 121, "descending", pols=("VV",)),  # single-pol: ignored
]


def test_groups_and_pairs():
    passes = group_passes(ITEMS, AOI_)
    assert len(passes) == 6
    pairs, notes = select_pairs(passes, date(2026, 8, 26))
    by = {p.relative_orbit: p for p in pairs}
    assert by[85].pre.start.day == 16 and by[85].post.start.day == 28
    assert by[85].pre2 is not None and by[85].pre2.start.day == 4
    # Track 19: the event-day pass is skipped without a time, so the pair is 08-14 → 09-07.
    assert by[19].pre.start.day == 14 and by[19].post.start.day == 7
    assert any("event day" in n for n in notes)
    # Ranking: track 85's post pass is closer to the event.
    assert pairs[0].relative_orbit == 85


def test_event_time_resolves_same_day():
    passes = group_passes(ITEMS, AOI_)
    pairs, _ = select_pairs(passes, datetime(2026, 8, 26, 3, 0, tzinfo=UTC))
    by = {p.relative_orbit: p for p in pairs}
    assert by[19].pre.start.day == 26  # 00:18 UTC is before a 03:00 UTC event


def test_low_coverage_excluded():
    small = {
        "type": "Polygon",
        "coordinates": [
            [[85.25, 27.95], [85.26, 27.95], [85.26, 27.96], [85.25, 27.96], [85.25, 27.95]]
        ],
    }
    items = [
        _item("2026-08-16T12:21:41Z", 85, geom=small),
        _item("2026-08-28T12:21:41Z", 85, geom=small),
    ]
    pairs, notes = select_pairs(group_passes(items, AOI_), date(2026, 8, 26))
    assert pairs == [] and any("covers" in n for n in notes)
