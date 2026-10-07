import json
import os
import warnings

import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from zarrpatch import (
    GridSampler,
    HDF5PatchDataset,
    ListSampler,
    ZarrPatchDataset,
    materialize,
    validate_materialized,
)

PATCH = {"time": 4, "lat": 16, "lon": 16}

# a handful of in-bounds corners for the ListSampler variants
LIST_CORNERS = [
    {"time": 0, "lat": 0, "lon": 0},
    {"time": 20, "lat": 44, "lon": 64},
    {"time": 7, "lat": 13, "lon": 33},
    {"time": 4, "lat": 8, "lon": 16},
]


def corners_key(corner_arr):
    return tuple(int(x) for x in corner_arr)


def live_samples(ds):
    """{corner_key: {var: array}} from one live iteration (return_corners on)."""
    ds.return_corners = True
    out = {}
    for p in ds:
        out[corners_key(p["corner"])] = {
            v: np.asarray(p[v]) for v in ds.variables
        }
    return out


def hdf5_samples(reader):
    out = {}
    for i in range(len(reader)):
        p = reader[i]
        out[corners_key(p["corner"])] = {
            v: np.asarray(p[v]) for v in p if v != "corner"
        }
    return out


def assert_roundtrip_equal(live, frozen):
    assert live.keys() == frozen.keys()
    for key in live:
        for var in live[key]:
            np.testing.assert_array_equal(frozen[key][var], live[key][var])


# --- 1. round-trip equality against the live dataset ------------------------


@pytest.mark.parametrize(
    "sampler_factory",
    [lambda: GridSampler(), lambda: ListSampler(LIST_CORNERS)],
    ids=["grid", "list"],
)
def test_roundtrip_equals_live(zarr3_sharded, tmp_path, sampler_factory):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=sampler_factory(),
        shuffle=True, shuffle_buffer=64,
    )
    live = live_samples(ds)

    path = str(tmp_path / "snap.h5")
    materialize(ds, path)

    frozen = hdf5_samples(HDF5PatchDataset(path))
    assert len(frozen) == len(ds)
    assert_roundtrip_equal(live, frozen)


def test_materialize_returns_path(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    assert materialize(ds, path) == path
    assert os.path.exists(path)


# --- 2. multi-worker read ---------------------------------------------------


def test_multiworker_read_yields_all_once(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=GridSampler())
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)

    reader = HDF5PatchDataset(path)
    n = len(reader)
    loader = DataLoader(reader, batch_size=None, num_workers=2)
    seen = [corners_key(p["corner"]) for p in loader]

    assert len(seen) == n
    assert len(set(seen)) == n
    assert set(seen) == set(hdf5_samples(reader).keys())


# --- 3. max_bytes refusal ---------------------------------------------------


def test_max_bytes_refuses_and_writes_nothing(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=GridSampler())
    path = str(tmp_path / "snap.h5")
    with pytest.raises(ValueError, match="max_bytes"):
        materialize(ds, path, max_bytes=1024)
    assert not os.path.exists(path)


def test_max_bytes_generous_budget_succeeds(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path, max_bytes=1 << 30)
    assert len(HDF5PatchDataset(path)) == len(ds)


# --- 4. overwrite semantics -------------------------------------------------


def test_overwrite_false_refuses_existing(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    with pytest.raises(FileExistsError):
        materialize(ds, path, overwrite=False)


def test_overwrite_true_replaces(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    # a different sample count so we can see the file actually changed
    ds2 = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS[:2]),
    )
    materialize(ds2, path, overwrite=True)
    assert len(HDF5PatchDataset(path)) == 2


# --- 5. compression options -------------------------------------------------


@pytest.mark.parametrize("compression", ["lzf", "gzip"])
def test_compression_roundtrips_equal_to_uncompressed(
    zarr3_sharded, tmp_path, compression
):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))

    plain = str(tmp_path / "plain.h5")
    comp = str(tmp_path / f"{compression}.h5")
    materialize(ds, plain)
    materialize(ds, comp, compression=compression)

    assert_roundtrip_equal(
        hdf5_samples(HDF5PatchDataset(plain)),
        hdf5_samples(HDF5PatchDataset(comp)),
    )
    # confirm the compression filter is actually recorded on the file
    with h5py.File(comp, "r") as f:
        assert f["sst"].compression == compression


# --- 6. corner dataset ordering & dtype -------------------------------------


def test_corner_columns_sorted_and_int64(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)

    sorted_dims = sorted(PATCH)  # ['lat', 'lon', 'time']
    with h5py.File(path, "r") as f:
        corner = f["corner"]
        assert corner.dtype == np.int64
        assert corner.shape == (len(LIST_CORNERS), len(PATCH))
        rows = {tuple(int(x) for x in row) for row in corner[:]}
    expected = {tuple(c[d] for d in sorted_dims) for c in LIST_CORNERS}
    assert rows == expected

    # the reader hands corners back in the same sorted-dim order
    reader = HDF5PatchDataset(path)
    got = {corners_key(reader[i]["corner"]) for i in range(len(reader))}
    assert got == expected


# --- 7. fingerprint attrs ---------------------------------------------------


def test_fingerprint_attrs(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=GridSampler())
    ds.set_epoch(3)
    n = len(ds)
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)

    fp = HDF5PatchDataset(path).fingerprint
    assert json.loads(fp["patch_size"]) == PATCH
    assert json.loads(fp["variables"]) == list(ds.variables)
    assert json.loads(fp["dim_slices"]) == {}
    assert int(fp["seed"]) == ds.seed
    assert int(fp["epoch"]) == 3
    assert int(fp["length"]) == n
    assert fp["zarrpatch_version"] == "0.1.0"


def test_corner_columns_property(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    reader = HDF5PatchDataset(path)
    assert reader.corner_columns == sorted(PATCH)

    # pre-corner_columns snapshots fall back to sorted(patch_size)
    with h5py.File(path, "a") as f:
        del f.attrs["corner_columns"]
    assert HDF5PatchDataset(path).corner_columns == sorted(PATCH)


def test_fingerprint_is_a_copy(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    reader = HDF5PatchDataset(path)
    fp = reader.fingerprint
    fp["length"] = 999
    assert reader.fingerprint["length"] != 999  # mutation didn't leak


# --- 8. transform on the reader ---------------------------------------------


def test_transform_applied_per_sample(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS))
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)

    plain = HDF5PatchDataset(path)
    transformed = HDF5PatchDataset(
        path, transform=lambda p: {"stacked": np.stack([p["sst"], p["ssh"]])}
    )
    assert len(transformed) == len(plain)
    for i in range(len(plain)):
        t = transformed[i]
        assert set(t) == {"stacked"}
        assert t["stacked"].shape == (2, 4, 16, 16)
        np.testing.assert_array_equal(t["stacked"][0], plain[i]["sst"])
        np.testing.assert_array_equal(t["stacked"][1], plain[i]["ssh"])


# --- 8b. materializing a transformed (re-keyed/reshaped) dataset -------------


def test_materialized_transform_roundtrip(zarr3_sharded, tmp_path):
    # a transform that re-keys and reshapes (raw vars -> one stacked input)
    # must freeze correctly: layout comes from the first yielded sample
    def assemble(p):
        return {"input": np.stack([p["sst"], p["ssh"]]), "corner": p["corner"]}

    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        transform=assemble,
    )
    path = str(tmp_path / "snap.h5")
    materialize(ds, path, transform_id="assemble-v1")

    reader = HDF5PatchDataset(path)
    assert len(reader) == len(LIST_CORNERS)
    ds.return_corners = True  # the transform needs "corner" when read live
    live = {corners_key(p["corner"]): p["input"] for p in ds}
    for i in range(len(reader)):
        p = reader[i]
        assert set(p) == {"input", "corner"}
        assert p["input"].shape == (2, 4, 16, 16)
        np.testing.assert_array_equal(p["input"], live[corners_key(p["corner"])])


def test_transform_dropping_corner_raises(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        transform=lambda p: {"sst": p["sst"]},
    )
    path = str(tmp_path / "snap.h5")
    with pytest.raises(ValueError, match="corner"):
        materialize(ds, path, transform_id="drops-corner-v1")
    assert not os.path.exists(path)
    assert ds.return_corners is False


def test_empty_transformed_dataset_writes_corner_only(zarr3_sharded, tmp_path):
    # sample shapes are unknowable with no samples and a transform; only the
    # (empty) corner dataset is written
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler([]), transform=lambda p: p,
    )
    path = str(tmp_path / "empty.h5")
    materialize(ds, path, transform_id="identity-v1")
    assert len(HDF5PatchDataset(path)) == 0
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"corner"}
        assert f["corner"].shape == (0, 3)


def test_max_bytes_measures_transformed_sample(zarr3_sharded, tmp_path):
    # the budget must reflect the (small) transformed sample, not the raw patch
    def shrink(p):
        return {"m": np.float32(p["sst"].mean()), "corner": p["corner"]}

    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        transform=shrink,
    )
    raw_patch_bytes = 4 * 16 * 16 * 4  # one raw sst patch, far over budget
    path = str(tmp_path / "small.h5")
    materialize(ds, path, max_bytes=raw_patch_bytes, transform_id="mean-v1")
    assert len(HDF5PatchDataset(path)) == len(LIST_CORNERS)

    with pytest.raises(ValueError, match="max_bytes"):
        materialize(
            ds, str(tmp_path / "tiny.h5"), max_bytes=8,
            transform_id="mean-v1",
        )


def test_transformed_materialization_requires_identity(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        transform=lambda p: p,
    )
    with pytest.raises(ValueError, match="transform_id"):
        materialize(ds, str(tmp_path / "snap.h5"))


def test_validate_materialized_and_versioned_metadata(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS)
    )
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    result = validate_materialized(path)
    assert result.valid
    assert result.metadata["length"] == len(LIST_CORNERS)
    with h5py.File(path, "r") as f:
        assert f.attrs["complete"]
        assert int(f.attrs["zarrpatch_format_version"]) == 1
        assert len(f.attrs["configuration_sha256"]) == 64


def test_reader_rejects_incomplete_snapshot(tmp_path):
    path = str(tmp_path / "partial.h5")
    with h5py.File(path, "w") as f:
        f.attrs["zarrpatch_format_version"] = 1
        f.attrs["complete"] = False
    result = validate_materialized(path)
    assert not result.valid
    assert any("incomplete" in error for error in result.errors)
    with pytest.raises(ValueError, match="incomplete"):
        HDF5PatchDataset(path)


def test_failed_overwrite_preserves_existing_snapshot(
    zarr3_sharded, tmp_path, monkeypatch
):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS)
    )
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    before = open(path, "rb").read()

    import importlib

    materialize_module = importlib.import_module("zarrpatch.materialize")

    def reject(*args, **kwargs):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(materialize_module, "validate_materialized", reject)
    with pytest.raises(ValueError, match="forced validation"):
        materialize(ds, path, overwrite=True)
    assert open(path, "rb").read() == before
    assert not list(tmp_path.glob(".snap.h5.*.tmp"))


def test_storage_impact_logged(zarr3_sharded, tmp_path, caplog):
    # the info log reports the snapshot size and the source read region always
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
    )
    with caplog.at_level("INFO", logger="zarrpatch"):
        materialize(ds, str(tmp_path / "snap.h5"))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("materialize:" in m and "source read region" in m for m in msgs)


def test_overlap_warns_small_list_does_not(zarr3_sharded, tmp_path):
    # a full overlapping grid scan writes more than the region it came from
    over = ZarrPatchDataset(
        zarr3_sharded, PATCH,
        sampler=GridSampler(stride={"time": 1, "lat": 4, "lon": 4}),
    )
    with pytest.warns(UserWarning, match="exceeds the source read region"):
        materialize(over, str(tmp_path / "over.h5"))

    # a handful of disjoint corners covers far less than the region: no warning
    small = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        materialize(small, str(tmp_path / "small.h5"))
    assert not any(
        "exceeds the source read region" in str(w.message) for w in caught
    )
    assert not os.path.exists(str(tmp_path / "tiny.h5"))


# --- 9. dim_slices round-trip -----------------------------------------------


def test_dim_slices_roundtrip(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"],
        sampler=ListSampler([{"time": 0, "lat": 0, "lon": 0},
                             {"time": 4, "lat": 44, "lon": 64}]),
        dim_slices={"time": slice(16, 24)},
    )
    live = live_samples(ds)

    path = str(tmp_path / "snap.h5")
    materialize(ds, path)

    reader = HDF5PatchDataset(path)
    assert_roundtrip_equal(live, hdf5_samples(reader))
    # the restriction is recorded in the fingerprint
    assert json.loads(reader.fingerprint["dim_slices"]) == {"time": [16, 24]}


def test_dim_slices_on_non_patched_dim_roundtrip(zarr3_sharded, tmp_path):
    # 'time' restricted but not patched -> frozen patch keeps the sliced extent
    ds = ZarrPatchDataset(
        zarr3_sharded, {"lat": 16, "lon": 16}, variables=["sst"],
        sampler=ListSampler([{"lat": 8, "lon": 8}]),
        dim_slices={"time": (2, 10)},
    )
    live = live_samples(ds)
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    reader = HDF5PatchDataset(path)
    (p,) = [reader[i] for i in range(len(reader))]
    assert p["sst"].shape == (8, 16, 16)
    assert_roundtrip_equal(live, hdf5_samples(reader))


# --- 10. code-path corners --------------------------------------------------


@pytest.mark.parametrize("initial", [False, True])
def test_return_corners_flag_restored(zarr3_sharded, tmp_path, initial):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        return_corners=initial,
    )
    materialize(ds, str(tmp_path / "snap.h5"))
    assert ds.return_corners is initial


def test_return_corners_flag_restored_on_error(zarr3_sharded, tmp_path):
    # max_bytes refusal happens before any mutation, but a mid-build error path
    # must still restore the flag; force one with an impossible target path.
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        return_corners=False,
    )
    bad = str(tmp_path / "no_such_dir" / "snap.h5")
    with pytest.raises((OSError, FileNotFoundError, RuntimeError)):
        materialize(ds, bad)
    assert ds.return_corners is False


def test_empty_dataset_roundtrips(zarr3_sharded, tmp_path):
    ds = ZarrPatchDataset(zarr3_sharded, PATCH, sampler=ListSampler([]))
    assert len(ds) == 0
    path = str(tmp_path / "empty.h5")
    materialize(ds, path)

    reader = HDF5PatchDataset(path)
    assert len(reader) == 0
    with h5py.File(path, "r") as f:
        assert f["sst"].shape == (0, 4, 16, 16)
        assert f["corner"].shape == (0, 3)
    assert int(reader.fingerprint["length"]) == 0


def test_materialize_forces_corners_without_return_corners(zarr3_sharded, tmp_path):
    # dataset built with return_corners=False must still get a corner dataset
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=ListSampler(LIST_CORNERS),
        return_corners=False,
    )
    path = str(tmp_path / "snap.h5")
    materialize(ds, path)
    reader = HDF5PatchDataset(path)
    p = reader[0]
    assert "corner" in p
    assert p["corner"].shape == (3,)


@pytest.mark.parametrize("rank", [0, 1])
def test_materialize_writes_whole_epoch_inside_ddp(zarr3_sharded, tmp_path,
                                                   monkeypatch, rank):
    import zarrpatch.dataset as dsmod
    from zarrpatch import RandomSampler

    ds = ZarrPatchDataset(zarr3_sharded, PATCH, variables=["sst"],
                          sampler=RandomSampler(101), shuffle=False)
    full = live_samples(ds)  # not distributed yet: the whole epoch
    ds.return_corners = False
    monkeypatch.setattr(dsmod, "_dist_info", lambda: (rank, 2))
    assert len(ds) == 50  # the live dataset now yields only this rank's share
    path = str(tmp_path / f"ddp_{rank}.h5")
    materialize(ds, path)
    frozen = hdf5_samples(HDF5PatchDataset(path))
    assert frozen.keys() == full.keys() and len(frozen) == 101
    for key, sample in full.items():
        np.testing.assert_array_equal(frozen[key]["sst"], sample["sst"])
    # the per-rank view is restored for training
    assert not ds._single_rank and len(ds) == 50


def test_materialize_restores_rank_split_after_failure(zarr3_sharded, tmp_path,
                                                       monkeypatch):
    import zarrpatch.dataset as dsmod

    ds = ZarrPatchDataset(zarr3_sharded, PATCH, variables=["sst"],
                          sampler=ListSampler(LIST_CORNERS))
    monkeypatch.setattr(dsmod, "_dist_info", lambda: (0, 2))
    with pytest.raises(ValueError, match="max_bytes"):
        materialize(ds, str(tmp_path / "x.h5"), max_bytes=1)
    assert not ds._single_rank and len(ds) == 2
