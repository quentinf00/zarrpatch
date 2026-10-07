import numpy as np
import pytest
from torch.utils.data import DataLoader

from zarrpatch import (
    GridSampler,
    ListSampler,
    RandomSampler,
    ZarrPatchDataset,
    group_corners_by_block,
    open_cube,
)

PATCH = {"time": 4, "lat": 16, "lon": 16}


def corners_key(corner_arr):
    return tuple(int(x) for x in corner_arr)


@pytest.fixture(params=["zarr3_sharded", "zarr3_plain", "zarr2_plain", "netcdf_file",
                        "kerchunk_refs", "zarr_zip"])
def cube_path(request):
    return request.getfixturevalue(request.param)


def test_block_shape_prefers_shards(zarr3_sharded, zarr3_plain):
    assert open_cube(zarr3_sharded).block_shape("sst") == (12, 40, 40)
    assert open_cube(zarr3_plain).block_shape("sst") == (6, 20, 20)


def test_patches_match_direct_slicing(cube_path, dataset):
    corners = [
        {"time": 0, "lat": 0, "lon": 0},
        {"time": 20, "lat": 44, "lon": 64},
        {"time": 7, "lat": 13, "lon": 33},
    ]
    ds = ZarrPatchDataset(
        cube_path, PATCH, sampler=ListSampler(corners),
        shuffle=False, return_corners=True,
    )
    seen = {}
    for patch in ds:
        seen[corners_key(patch["corner"])] = patch
    assert len(seen) == len(corners)
    for c in corners:
        patch = seen[(c["lat"], c["lon"], c["time"])]  # sorted dim order
        for var in ("sst", "ssh"):
            expected = dataset[var].values[
                c["time"]:c["time"] + 4,
                c["lat"]:c["lat"] + 16,
                c["lon"]:c["lon"] + 16,
            ]
            np.testing.assert_array_equal(patch[var], expected)


def test_variable_without_patch_dim_is_read_full_when_requested(zarr3_sharded, dataset):
    # mask has no time dim: patch on (lat, lon) only, request all three vars
    ds = ZarrPatchDataset(
        zarr3_sharded, {"lat": 16, "lon": 16},
        variables=["sst", "mask"],
        sampler=ListSampler([{"lat": 8, "lon": 8}]),
        shuffle=False,
    )
    (patch,) = list(ds)
    assert patch["sst"].shape == (24, 16, 16)  # full time extent
    np.testing.assert_array_equal(
        patch["mask"], dataset["mask"].values[8:24, 8:24]
    )


def test_auto_variable_selection_excludes_missing_dims(zarr3_sharded):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, shuffle=False)
    assert ds.variables == ["ssh", "sst"]  # mask lacks the time dim


def test_grid_epoch_is_exact_partition(zarr3_sharded):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=GridSampler(),
        shuffle=True, shuffle_buffer=64, return_corners=True,
    )
    expected = {corners_key(p["corner"])
                for p in ZarrPatchDataset(
                    zarr3_sharded, PATCH, sampler=GridSampler(),
                    shuffle=False, return_corners=True)}
    got = [corners_key(p["corner"]) for p in ds]
    assert len(got) == len(ds)
    assert set(got) == expected
    assert len(set(got)) == len(got)


def test_multiworker_yields_each_patch_once(zarr3_sharded):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(100),
        shuffle=True, shuffle_buffer=32, return_corners=True,
    )
    single = [corners_key(p["corner"]) for p in ds]
    loader = DataLoader(ds, batch_size=None, num_workers=2)
    multi = [corners_key(p["corner"]) for p in loader]
    assert sorted(multi) == sorted(single)
    assert len(multi) == 100


def test_set_epoch_changes_random_draw(zarr3_sharded):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(50), return_corners=True,
    )
    e0 = [corners_key(p["corner"]) for p in ds]
    ds.set_epoch(1)
    e1 = [corners_key(p["corner"]) for p in ds]
    assert e0 != e1
    ds.set_epoch(0)
    assert sorted(corners_key(p["corner"]) for p in ds) == sorted(e0)


def test_shuffle_changes_order_but_not_content(zarr3_sharded):
    kwargs = dict(sampler=GridSampler(), return_corners=True)
    ordered = [corners_key(p["corner"]) for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, shuffle=False, **kwargs)]
    shuffled_ = [corners_key(p["corner"]) for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, shuffle=True, shuffle_buffer=128, **kwargs)]
    assert ordered != shuffled_
    assert sorted(ordered) == sorted(shuffled_)


def test_naive_mode_matches_grouped(zarr3_sharded):
    kwargs = dict(sampler=RandomSampler(40), shuffle=False, return_corners=True)
    grouped = {corners_key(p["corner"]): p["sst"] for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, group_by_block=True, **kwargs)}
    naive = {corners_key(p["corner"]): p["sst"] for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, group_by_block=False, **kwargs)}
    assert grouped.keys() == naive.keys()
    for k in grouped:
        np.testing.assert_array_equal(grouped[k], naive[k])


def test_dataloader_batching_and_transform(zarr3_sharded):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(32),
        transform=lambda p: {"x": np.stack([p["sst"], p["ssh"]])},
    )
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert tuple(batch["x"].shape) == (8, 2, 4, 16, 16)


def test_group_corners_by_block():
    corners = [{"x": 0}, {"x": 9}, {"x": 10}, {"x": 25}]
    groups = group_corners_by_block(corners, {"x": 10})
    assert groups == [[{"x": 0}, {"x": 9}], [{"x": 10}], [{"x": 25}]]


def test_zarr_variables_exclude_coordinate_arrays(zarr3_plain, zarr2_plain):
    assert open_cube(zarr3_plain).variables == ["mask", "ssh", "sst"]
    assert open_cube(zarr2_plain).variables == ["mask", "ssh", "sst"]


def test_1d_patch_does_not_select_coord_variable(zarr3_plain):
    ds = ZarrPatchDataset(zarr3_plain, {"time": 4}, shuffle=False)
    assert ds.variables == ["ssh", "sst"]


def test_set_epoch_reaches_persistent_workers(zarr3_sharded):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(30), return_corners=True,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=1,
                        persistent_workers=True)
    e0 = sorted(corners_key(p["corner"]) for p in loader)
    ds.set_epoch(1)
    e1 = sorted(corners_key(p["corner"]) for p in loader)
    assert e0 != e1
    # workers draw the same corners as a single-process epoch-1 iteration
    ds_ref = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(30), return_corners=True,
    )
    ds_ref.set_epoch(1)
    assert e1 == sorted(corners_key(p["corner"]) for p in ds_ref)


def _per_rank(ds, monkeypatch, world):
    import zarrpatch.dataset as dsmod

    out = []
    for rank in range(world):
        monkeypatch.setattr(dsmod, "_dist_info", lambda r=rank: (r, world))
        out.append(([corners_key(p["corner"]) for p in ds], len(ds)))
    monkeypatch.undo()
    return out


def test_ddp_ranks_partition_epoch(zarr3_sharded, monkeypatch):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(100), return_corners=True,
    )
    full = sorted(corners_key(p["corner"]) for p in ds)
    per_rank = _per_rank(ds, monkeypatch, 2)
    assert sorted(per_rank[0][0] + per_rank[1][0]) == full
    # exactly equal, not merely balanced: unequal ranks deadlock DDP
    assert [len(c) for c, _ in per_rank] == [50, 50]
    assert [n for _, n in per_rank] == [50, 50]  # len() is per rank


@pytest.mark.parametrize("remainder, per_rank_n", [("drop", 33), ("pad", 34)])
def test_ddp_remainder(zarr3_sharded, monkeypatch, remainder, per_rank_n):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(101), return_corners=True,
        ddp_remainder=remainder,
    )
    full = [corners_key(p["corner"]) for p in ds]
    assert len(ds) == 101 and ds.epoch_size() == 101
    per_rank = _per_rank(ds, monkeypatch, 3)
    assert [len(c) for c, _ in per_rank] == [per_rank_n] * 3
    assert [n for _, n in per_rank] == [per_rank_n] * 3
    got = [c for cs, _ in per_rank for c in cs]
    if remainder == "drop":  # a subset of the epoch, each patch at most once
        assert len(set(got)) == len(got) == 99
        assert set(got) <= set(full)
    else:  # the whole epoch, plus two re-yielded patches
        assert set(got) == set(full) and len(got) == 102


def test_ddp_per_worker_counts_identical_across_ranks():
    from zarrpatch.dataset import _shard_groups

    rng = np.random.default_rng(0)
    groups = [[{"i": j}] * int(n) for j, n in enumerate(rng.integers(1, 30, 57))]
    total = sum(map(len, groups))
    for world, nw in [(2, 3), (4, 5), (3, 1)]:
        counts = [[sum(map(len, _shard_groups(groups, r, world, w, nw, "drop")))
                   for w in range(nw)] for r in range(world)]
        assert all(c == counts[0] for c in counts)  # same batches per rank
        assert max(counts[0]) - min(counts[0]) <= 1
        assert sum(map(sum, counts)) == total // world * world


def test_ddp_remainder_validation(zarr3_sharded):
    with pytest.raises(ValueError, match="ddp_remainder"):
        ZarrPatchDataset(zarr3_sharded, PATCH, ddp_remainder="bogus")


def test_prefetch_matches_no_prefetch(zarr3_sharded):
    kwargs = dict(sampler=RandomSampler(60), shuffle=False, return_corners=True)
    ref = {corners_key(p["corner"]): p["sst"] for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, prefetch=0, **kwargs)}
    pre = {corners_key(p["corner"]): p["sst"] for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, prefetch=2, **kwargs)}
    assert ref.keys() == pre.keys()
    for k in ref:
        np.testing.assert_array_equal(ref[k], pre[k])


def test_auto_shuffle_buffer(zarr3_sharded):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(80),
        shuffle_buffer="auto", return_corners=True,
    )
    got = [corners_key(p["corner"]) for p in ds]
    assert len(got) == 80
    with pytest.raises(ValueError):
        ZarrPatchDataset(zarr3_sharded, PATCH, shuffle_buffer="bogus")


def test_symbolic_shuffle_buffer_sizes():
    from zarrpatch.dataset import DEFAULT_SHUFFLE_BUFFER, resolve_shuffle_buffer

    sparse = [[{}] * 3, [{}] * 12]
    dense = [[{}] * 1000]
    assert resolve_shuffle_buffer("io", sparse) == 48
    # "auto" is never weaker than the default ("io" alone can be far too small to mix batches)
    assert resolve_shuffle_buffer("auto", sparse) == DEFAULT_SHUFFLE_BUFFER
    assert resolve_shuffle_buffer("auto", dense) == 4000
    assert resolve_shuffle_buffer(64, sparse) == 64


def test_group_block_matches_direct_slicing(zarr3_plain, dataset):
    ds = ZarrPatchDataset(
        zarr3_plain, PATCH, variables=["sst"], sampler=RandomSampler(60),
        group_block={"time": 12, "lat": 40, "lon": 60}, return_corners=True,
    )
    assert ds._block_size == {"time": 12, "lat": 40, "lon": 60}
    n = 0
    for p in ds:
        la, lo, t = (int(x) for x in p["corner"])  # sorted(patch_size) order
        ref = dataset.sst.values[t:t + 4, la:la + 16, lo:lo + 16]
        np.testing.assert_array_equal(p["sst"], ref)
        n += 1
    assert n == 60


def test_group_block_validation(zarr3_plain):
    with pytest.raises(ValueError, match="multiple of the chunk edge"):
        ZarrPatchDataset(zarr3_plain, PATCH, group_block={"lat": 30})
    with pytest.raises(ValueError, match="not a patch dim"):
        ZarrPatchDataset(zarr3_plain, PATCH, group_block={"depth": 20})


def test_chunk_straddle_warning(zarr3_plain):
    with pytest.warns(UserWarning, match="straddles the next chunk"):
        ZarrPatchDataset(zarr3_plain, PATCH, variables=["sst"])
    import warnings as w

    with w.catch_warnings():
        w.simplefilter("error")
        ZarrPatchDataset(zarr3_plain, PATCH, variables=["sst"],
                         group_block={"time": 12, "lat": 60, "lon": 80})


def test_contiguous_netcdf_blocks_are_capped(dataset, tmp_path):
    path = tmp_path / "contiguous.nc"
    dataset[["sst"]].to_netcdf(path)  # no chunking -> contiguous storage
    with pytest.warns(UserWarning, match="virtual blocks"):
        ds = ZarrPatchDataset(
            str(path), PATCH, sampler=ListSampler([
                {"time": 0, "lat": 0, "lon": 0},
                {"time": 20, "lat": 44, "lon": 64},
            ]),
            shuffle=False, return_corners=True, max_block_bytes=100_000,
        )
    # virtual blocks, not the whole cube
    assert any(ds._block_size[d] < n for d, n in ds._dim_sizes.items())
    for patch in ds:
        c = dict(zip(sorted(PATCH), (int(x) for x in patch["corner"])))
        expected = dataset["sst"].values[
            c["time"]:c["time"] + 4,
            c["lat"]:c["lat"] + 16,
            c["lon"]:c["lon"] + 16,
        ]
        np.testing.assert_array_equal(patch["sst"], expected)


def test_dim_slices_patches_match_sliced_cube(zarr3_sharded, dataset):
    val = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"],
        sampler=ListSampler([{"time": 0, "lat": 0, "lon": 0},
                             {"time": 4, "lat": 44, "lon": 64}]),
        shuffle=False, return_corners=True, dim_slices={"time": slice(16, 24)},
    )
    n = 0
    for p in val:
        c = dict(zip(sorted(PATCH), (int(x) for x in p["corner"])))
        expected = dataset["sst"].values[
            16 + c["time"]:16 + c["time"] + 4,
            c["lat"]:c["lat"] + 16,
            c["lon"]:c["lon"] + 16,
        ]
        np.testing.assert_array_equal(p["sst"], expected)
        n += 1
    assert n == 2


def test_dim_slices_random_sampling_stays_in_slice(zarr3_sharded, dataset):
    # offsets (5, 10) are deliberately misaligned with the storage blocks
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"],
        sampler=RandomSampler(60), return_corners=True,
        dim_slices={"time": (5, 21), "lat": (10, 50)},
    )
    n = 0
    for p in ds:
        c = dict(zip(sorted(PATCH), (int(x) for x in p["corner"])))
        assert 0 <= c["time"] <= 16 - 4 and 0 <= c["lat"] <= 40 - 16
        expected = dataset["sst"].values[
            5 + c["time"]:5 + c["time"] + 4,
            10 + c["lat"]:10 + c["lat"] + 16,
            c["lon"]:c["lon"] + 16,
        ]
        np.testing.assert_array_equal(p["sst"], expected)
        n += 1
    assert n == 60


def test_dim_slices_on_non_patched_dim(zarr3_sharded, dataset):
    ds = ZarrPatchDataset(
        zarr3_sharded, {"lat": 16, "lon": 16}, variables=["sst"],
        sampler=ListSampler([{"lat": 8, "lon": 8}]), shuffle=False,
        dim_slices={"time": (2, 10)},
    )
    (p,) = list(ds)
    assert p["sst"].shape == (8, 16, 16)
    np.testing.assert_array_equal(
        p["sst"], dataset["sst"].values[2:10, 8:24, 8:24]
    )


def test_dim_slices_validation(zarr3_sharded):
    with pytest.raises(ValueError, match="out of bounds"):
        ZarrPatchDataset(zarr3_sharded, PATCH, dim_slices={"time": (0, 99)})
    with pytest.raises(ValueError, match="step not supported"):
        ZarrPatchDataset(zarr3_sharded, PATCH,
                         dim_slices={"time": slice(0, 8, 2)})
    with pytest.raises(ValueError, match="not in the dims"):
        ZarrPatchDataset(zarr3_sharded, PATCH, dim_slices={"depth": (0, 4)})
    with pytest.raises(ValueError, match="exceeds dim"):
        # patch (time: 4) no longer fits in the 3-step slice
        list(ZarrPatchDataset(zarr3_sharded, PATCH,
                              dim_slices={"time": (0, 3)}))


def test_open_cube_zip_backend(zarr_zip, zarr3_plain):
    from zarrpatch import ZarrReader

    r = open_cube(zarr_zip)
    assert isinstance(r, ZarrReader)
    ref = open_cube(zarr3_plain)
    assert r.variables == ref.variables
    for v in r.variables:
        assert r.dims(v) == ref.dims(v)
        assert r.block_shape(v) == ref.block_shape(v)


def test_zip_reader_picklable(zarr_zip):
    import pickle

    r = open_cube(zarr_zip)
    region = {"time": slice(0, 4), "lat": slice(0, 16), "lon": slice(0, 16)}
    before = r.read("sst", region)
    r2 = pickle.loads(pickle.dumps(r))
    np.testing.assert_array_equal(r2.read("sst", region), before)


def test_dataset_accepts_reader_instance(zarr3_sharded, dataset):
    reader = open_cube(zarr3_sharded)
    ds = ZarrPatchDataset(
        reader, PATCH, sampler=ListSampler([{"time": 0, "lat": 0, "lon": 0}]),
        shuffle=False,
    )
    (patch,) = list(ds)
    np.testing.assert_array_equal(
        patch["sst"], dataset["sst"].values[0:4, 0:16, 0:16]
    )


def test_warns_on_mismatched_block_shapes(dataset, tmp_path):
    path = tmp_path / "mixed.zarr"
    dataset[["sst", "ssh"]].to_zarr(
        path, zarr_format=3, mode="w",
        encoding={"sst": {"chunks": (6, 20, 20)},
                  "ssh": {"chunks": (12, 20, 20)}},
    )
    with pytest.warns(UserWarning, match="block sizes"):
        ZarrPatchDataset(str(path), PATCH, variables=["sst", "ssh"])
