import json

from dhara.jobs import pipeline as P


def test_cached_pass_from_another_chain_is_not_used(tmp_path):
    own = tmp_path / "s1_x_GAMMA0_TERRAIN_grd.tif"
    other = tmp_path / "s1_x_GAMMA0_TERRAIN.tif"
    other.write_bytes(b"x")
    own.symlink_to(other.name)
    assert not P._usable_pass(own) and not own.is_symlink()  # link removed, pass reprocessed
    own.write_bytes(b"x")
    assert not P._usable_pass(own)  # no sidecar: not written by this pipeline
    own.write_bytes(b"x")
    own.with_suffix(".json").write_text(json.dumps({"route": "grd"}))
    assert P._usable_pass(own)
    assert not P._usable_pass(tmp_path / "missing.tif")
