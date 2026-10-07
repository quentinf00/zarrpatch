import json

import numpy as np
import pytest

from zarrpatch import ListSampler, ZarrPatchDataset, analyze_layout


PATCH = {"time": 4, "lat": 16, "lon": 16}
CORNERS = [
    {"time": 0, "lat": 0, "lon": 0},
    {"time": 1, "lat": 1, "lon": 1},
    {"time": 20, "lat": 44, "lon": 64},
]


def test_plan_is_exact_deterministic_json_and_read_free(zarr3_sharded, monkeypatch):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"],
        sampler=ListSampler(CORNERS), shuffle_buffer="auto",
    )

    def payload_read_forbidden(*args, **kwargs):
        raise AssertionError("planning read array payload")

    monkeypatch.setattr(ds.reader, "read", payload_read_forbidden)
    first = ds.plan(world_size=2, num_workers=2)
    second = ds.plan(world_size=2, num_workers=2)
    assert first == second
    assert first.exact
    assert first.patches == 3
    assert first.groups == 2
    # 3 patches over 2 ranks: the remainder is dropped so ranks stay equal
    assert first.yielded_patches == 2
    assert sum(first.shard_patch_counts) == 2
    assert first.shard_patch_counts[:2] == first.shard_patch_counts[2:]
    assert first.logical_bytes == first.logical_patch_bytes * 2
    assert first.decoded_bytes > 0
    assert json.loads(first.to_json())["patches"] == 3
    assert "3 patches in 2 physical groups" in first.summary()


def test_analyze_layout_matches_dataset_plan(zarr3_sharded):
    kwargs = {
        "variables": ["sst"],
        "sampler": ListSampler(CORNERS),
        "shuffle": False,
    }
    direct = ZarrPatchDataset(zarr3_sharded, PATCH, **kwargs).plan()
    analyzed = analyze_layout(zarr3_sharded, PATCH, **kwargs)
    assert analyzed == direct


def test_plan_reports_incompatible_variable_layout(dataset, tmp_path):
    path = tmp_path / "mixed.zarr"
    dataset[["sst", "ssh"]].to_zarr(
        path, zarr_format=3, mode="w",
        encoding={"sst": {"chunks": (6, 20, 20)},
                  "ssh": {"chunks": (12, 20, 20)}},
    )
    with pytest.warns(UserWarning, match="block sizes"):
        ds = ZarrPatchDataset(
            str(path), PATCH, variables=["sst", "ssh"],
            sampler=ListSampler(CORNERS),
        )
    plan = ds.plan()
    assert plan.reference_variable == "sst"
    assert plan.incompatible_variables == ["ssh"]
    assert any("do not match" in warning for warning in plan.warnings)


def test_plan_validates_topology(zarr3_sharded):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, variables=["sst"])
    with pytest.raises(ValueError, match="must be >= 1"):
        ds.plan(num_workers=0)


def _block_key(corner, block):
    la, lo, t = corner  # sorted(PATCH) order
    return (t // block["time"], la // block["lat"], lo // block["lon"])


@pytest.mark.parametrize("buffer", ["io", 256])
def test_batch_mixing_replays_the_real_stream(zarr3_sharded, buffer):
    from torch.utils.data import DataLoader

    from zarrpatch import RandomSampler

    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"], sampler=RandomSampler(300),
        shuffle_buffer=buffer, return_corners=True, seed=3,
    )
    mixing = ds.plan(batch_size=16).batch_mixing
    distinct = []
    for batch in DataLoader(ds, batch_size=16):
        keys = {_block_key(c.tolist(), ds._block_size) for c in batch["corner"]}
        distinct.append(len(keys))
    assert mixing["batches"] == len(distinct)
    assert mixing["distinct_groups_mean"] == pytest.approx(np.mean(distinct))
    assert mixing["distinct_groups_min"] == min(distinct)


def test_small_buffer_is_flagged_as_poor_mixing(zarr3_sharded):
    from zarrpatch import RandomSampler

    kwargs = dict(variables=["sst"], sampler=RandomSampler(2000))
    io = ZarrPatchDataset(zarr3_sharded, PATCH, shuffle_buffer=2, **kwargs)
    auto = ZarrPatchDataset(zarr3_sharded, PATCH, shuffle_buffer="auto", **kwargs)
    bad = io.plan(batch_size=32)
    good = auto.plan(batch_size=32)
    assert any("poor batch mixing" in w for w in bad.warnings)
    assert not any("poor batch mixing" in w for w in good.warnings)
    assert (good.batch_mixing["distinct_groups_mean"]
            > bad.batch_mixing["distinct_groups_mean"])


def test_decoded_bytes_are_chunk_aligned_and_group_block_reduces_them(zarr3_plain):
    from zarrpatch import RandomSampler

    kwargs = dict(variables=["sst"], sampler=RandomSampler(2000), shuffle=False)
    with pytest.warns(UserWarning, match="straddles"):
        by_chunk = ZarrPatchDataset(zarr3_plain, PATCH, **kwargs).plan()
    grouped = ZarrPatchDataset(
        zarr3_plain, PATCH, group_block={"time": 24, "lat": 60, "lon": 80},
        **kwargs,
    ).plan()
    assert by_chunk.decoded_bytes >= by_chunk.region_bytes
    assert grouped.decoded_bytes < by_chunk.decoded_bytes / 2
    assert grouped.grouping_block == {"time": 24, "lat": 60, "lon": 80}
