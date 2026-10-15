from datetime import date

from dhara.adapters.sentinel2 import S2Scene, choose_pair


def _s(day: str, aoi: float, reach: float | None = None) -> S2Scene:
    return S2Scene(
        day=date.fromisoformat(day),
        scene_ids=[day],
        aoi_cloud_fraction=aoi,
        reach_cloud_fraction=reach,
    )


def test_partly_clear_valley_scene_is_usable_but_ranking_is_unchanged():
    ev = date(2026, 8, 26)
    # Whole-AOI cloud 74 %, valley 61 %: rejected by the AOI rule, usable by the valley rule.
    cands = [_s("2026-08-12", 0.04, 0.01), _s("2026-08-27", 0.74, 0.61), _s("2026-08-29", 1.0, 1.0)]
    p = choose_pair(cands, ev)
    assert p.pre and p.pre.day == date(2026, 8, 12)
    assert p.post and p.post.day == date(2026, 8, 27)
    assert choose_pair([_s("2026-08-27", 0.74)], ev).post is None  # no valley measure: AOI rule
    # Ranking stays on AOI cloud: the clearer whole-AOI scene wins even if its valley is cloudier.
    p2 = choose_pair([_s("2026-08-27", 0.20, 0.30), _s("2026-08-28", 0.40, 0.10)], ev)
    assert p2.post and p2.post.day == date(2026, 8, 27)
