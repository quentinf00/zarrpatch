from math import prod

import pytest

from zarrpatch.guidelines import MiB, recommend_layout

DIMS = {"time": 2000, "lat": 2400, "lon": 3600}
PATCH = {"lat": 128, "lon": 128}


def test_random_sampling_chunk_matches_patch():
    rec = recommend_layout(DIMS, PATCH, sampling="random")
    assert rec.chunks["lat"] == 128
    assert rec.chunks["lon"] == 128
    assert rec.chunk_bytes <= 16 * MiB


def test_shards_are_chunk_multiples_and_target_size():
    rec = recommend_layout(DIMS, PATCH, target_shard_mib=256)
    assert rec.shards is not None
    for d in DIMS:
        assert rec.shards[d] % rec.chunks[d] == 0
        assert rec.shards[d] >= rec.chunks[d]
    assert rec.shard_bytes >= 64 * MiB
    assert rec.n_files < prod(
        -(-DIMS[d] // rec.chunks[d]) for d in DIMS
    )  # fewer files than unsharded


def test_no_shards_reports_file_count():
    rec = recommend_layout(DIMS, PATCH, use_shards=False)
    assert rec.shards is None
    expected = prod(-(-DIMS[d] // rec.chunks[d]) for d in DIMS)
    assert rec.n_files == expected


def test_grid_sampling_aligns_to_stride():
    rec = recommend_layout(
        DIMS, PATCH, sampling="grid", stride={"lat": 64, "lon": 64}
    )
    assert rec.chunks["lat"] % 64 == 0
    assert rec.chunks["lon"] % 64 == 0
    assert rec.chunks["lat"] >= 128


def test_unpatched_dim_shrunk_to_fit_target():
    rec = recommend_layout(DIMS, PATCH, target_chunk_mib=4.0)
    # full time extent would be 2000*128*128*4 bytes = 125 GiB; must shrink
    assert rec.chunks["time"] < DIMS["time"]
    assert rec.chunk_bytes <= 8 * MiB


def test_chunk_capped_at_dim_size():
    rec = recommend_layout({"lat": 100, "lon": 100}, {"lat": 128, "lon": 64})
    assert rec.chunks["lat"] == 100


def test_invalid_sampling_raises():
    with pytest.raises(ValueError):
        recommend_layout(DIMS, PATCH, sampling="stochastic")


def test_summary_is_printable():
    text = recommend_layout(DIMS, PATCH).summary()
    assert "chunks" in text and "shards" in text


def test_unsharded_recommends_multi_chunk_group_block():
    rec = recommend_layout(DIMS, PATCH, sampling="random", use_shards=False)
    assert rec.group_block is not None
    for d, edge in rec.group_block.items():
        assert edge % rec.chunks[d] == 0
        # <= 25% straddle: the halo (patch - 1) is at most a quarter of the edge
        assert (PATCH[d] - 1) / edge <= 0.25
    assert "group_block" in rec.summary()
    assert not any("waste little" in n for n in rec.notes)


def test_sharded_leaves_grouping_to_shards():
    assert recommend_layout(DIMS, PATCH).group_block is None
