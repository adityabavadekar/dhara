"""The mountain hold-out fold must never leak into train/val, even on the tile-split fallback."""

from typing import cast

import pytest

pytest.importorskip("torch")

from training.train import make_splits  # noqa: E402


def test_fallback_split_keeps_mountain_out_of_training():
    # Seven events, all listed as train events: the event-level val split is empty, so the
    # tile-level fallback runs. This is the Kuro Siwo shard the deployed model was trained on.
    events = {470: 14.5, 492: 29.8, 496: 49.3, 497: 11.0, 498: 161.2, 502: 59.4, 518: 26.1}
    rows = [
        {"actid": a, "relief_m": r, "valid_frac": 1.0} for a, r in events.items() for _ in range(10)
    ]
    cfg = {
        "splits": {
            "mountain_holdout_events": 4,
            "train_acts": list(events),
            "val_acts": [],
            "test_acts": [],
        }
    }
    s = make_splits(rows, cfg)
    mountain = set(s["mountain"])
    assert mountain
    assert not mountain & set(s["train"])
    assert not mountain & set(s["val"])
    meta = cast(dict, s["_meta"])  # make_splits packs a "_meta" dict alongside the index lists
    assert meta["tile_split_fallback"] is True
