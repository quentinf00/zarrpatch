import pickle
from collections import Counter

import numpy as np
import pytest
import xarray as xr

from zarrpatch import (
    open_cube,
    GridSampler,
    MultiCubeReader,
    WeightedSampler,
    ZarrPatchDataset,
    expand_units,
    group_corners_by_block,
)

LAT, LON = 60, 80
PATCH = {"store": 1, "lat": 16, "lon": 16}


def _write(ds, path, chunks=(1, 20, 20)):
    enc = {v: {"chunks": chunks} for v in ds.data_vars}
    ds.to_zarr(path, zarr_format=3, encoding=enc, mode="w")
    return str(path)


def test_expand_units(multi_stores, monkeypatch):
    import zarrpatch.readers as rmod

    paths, arrays = multi_stores
    orig = rmod.open_cube
    calls = []
    monkeypatch.setattr(
        rmod, "open_cube",
        lambda p, b=None: (calls.append(p), orig(p, b))[1],
    )
    units = expand_units(paths, "member")
    assert len(units) == sum(a.shape[0] for a in arrays)
    assert [u[1]["member"] for u in units] == [0, 0, 1, 2, 0, 1]
    assert calls == list(paths)  # each store opened exactly once


def test_multicube_shape_and_blockshape(multi_stores):
    paths, arrays = multi_stores
    reader = MultiCubeReader(expand_units(paths, "member"))
    n_units = sum(a.shape[0] for a in arrays)
    assert reader.dims("u10") == ("store", "lat", "lon")
    assert reader.shape("u10") == (n_units, LAT, LON)
    assert reader.block_shape("u10") == (1, 20, 20)
    assert reader.variables == ["u10", "v10"]


def test_multicube_variable_subset(multi_stores):
    paths, _ = multi_stores
    reader = MultiCubeReader(
        expand_units(paths, "member"), variables=["v10"], cache_size=len(paths)
    )
    assert reader.variables == ["v10"]
    assert reader.dims("v10") == ("store", "lat", "lon")
    with pytest.raises(ValueError, match="missing requested variables"):
        MultiCubeReader(expand_units(paths, "member"), variables=["missing"])


def test_multicube_read_matches_source(multi_stores):
    paths, arrays = multi_stores
    reader = MultiCubeReader(expand_units(paths, "member"))
    expected = np.concatenate(arrays, axis=0)  # (n_units, lat, lon)
    for u in (0, 3, 5):
        region = {"store": slice(u, u + 1), "lat": slice(5, 25), "lon": slice(0, 16)}
        got = reader.read("u10", region)
        assert got.shape == (1, 20, 16)
        np.testing.assert_array_equal(got[0], expected[u, 5:25, 0:16])
        # v10 == u10 * 2 in the fixture
        np.testing.assert_array_equal(
            reader.read("v10", region)[0], expected[u, 5:25, 0:16] * 2
        )


def test_multicube_schema_mismatch(multi_stores, tmp_path):
    paths, _ = multi_stores
    good = expand_units(paths, "member")

    # differing (non-virtual) shape -> ValueError on first read of that unit
    small = xr.Dataset(
        {"u10": (["member", "lat", "lon"], np.zeros((1, LAT - 1, LON), "float32")),
         "v10": (["member", "lat", "lon"], np.zeros((1, LAT - 1, LON), "float32"))},
    )
    bad_shape = _write(small, tmp_path / "bad_shape.zarr")
    r = MultiCubeReader(good + [(bad_shape, {"member": 0})])
    with pytest.raises(ValueError, match="shape"):
        r.read("u10", {"store": slice(len(good), len(good) + 1)})

    # differing dtype -> ValueError
    dt = xr.Dataset(
        {"u10": (["member", "lat", "lon"], np.zeros((1, LAT, LON), "float64")),
         "v10": (["member", "lat", "lon"], np.zeros((1, LAT, LON), "float64"))},
    )
    bad_dtype = _write(dt, tmp_path / "bad_dtype.zarr")
    r = MultiCubeReader(good + [(bad_dtype, {"member": 0})])
    with pytest.raises(ValueError, match="dtype"):
        r.read("u10", {"store": slice(len(good), len(good) + 1)})

    # differing chunks -> warn (reads still work)
    ch = xr.Dataset(
        {"u10": (["member", "lat", "lon"], np.zeros((1, LAT, LON), "float32")),
         "v10": (["member", "lat", "lon"], np.zeros((1, LAT, LON), "float32"))},
    )
    bad_chunk = _write(ch, tmp_path / "bad_chunk.zarr", chunks=(1, 10, 10))
    r = MultiCubeReader(good + [(bad_chunk, {"member": 0})])
    with pytest.warns(UserWarning, match="block_shape"):
        out = r.read("u10", {"store": slice(len(good), len(good) + 1)})
    assert out.shape == (1, LAT, LON)


def test_multicube_dataset_end_to_end(multi_stores):
    paths, arrays = multi_stores
    reader = MultiCubeReader(expand_units(paths, "member"))
    expected = np.concatenate(arrays, axis=0)
    ds = ZarrPatchDataset(
        reader, PATCH, variables=["u10"], sampler=GridSampler(),
        shuffle=False, return_corners=True,
    )
    n = 0
    for p in ds:
        lat, lon, store = (int(x) for x in p["corner"])  # sorted dim order
        assert p["u10"].shape == (1, 16, 16)
        np.testing.assert_array_equal(
            p["u10"][0], expected[store, lat:lat + 16, lon:lon + 16]
        )
        n += 1
    assert n > 0

    # block groups never cross stores
    corners = GridSampler().sample(ds._dim_sizes, ds.patch_size,
                                   np.random.default_rng(0))
    for group in group_corners_by_block(corners, ds._block_size):
        assert len({c["store"] for c in group}) == 1


def test_multicube_weighted_store(multi_stores):
    paths, arrays = multi_stores
    n_units = sum(a.shape[0] for a in arrays)
    weight = np.ones(n_units)
    weight[3] = 100.0
    reader = MultiCubeReader(expand_units(paths, "member"))
    ds = ZarrPatchDataset(
        reader, PATCH, variables=["u10"],
        sampler=WeightedSampler(2000, weight, ["store"]),
        shuffle=False, return_corners=True,
    )
    counts = Counter(int(p["corner"][2]) for p in ds)
    assert counts[3] > 1500  # the heavily-weighted store dominates


def test_multicube_requires_unit_virtual_patch(multi_stores):
    paths, _ = multi_stores
    reader = MultiCubeReader(expand_units(paths, "member"))
    with pytest.raises(ValueError, match="must be 1"):
        ZarrPatchDataset(reader, {"store": 2, "lat": 16, "lon": 16},
                         variables=["u10"])
    with pytest.raises(ValueError, match="virtual dim"):
        ZarrPatchDataset(reader, {"lat": 16, "lon": 16}, variables=["u10"])


def test_multicube_picklable(multi_stores):
    paths, arrays = multi_stores
    reader = MultiCubeReader(expand_units(paths, "member"))
    region = {"store": slice(2, 3), "lat": slice(0, 16), "lon": slice(0, 16)}
    before = reader.read("u10", region)
    r2 = pickle.loads(pickle.dumps(reader))
    assert r2._cache is None  # cache dropped on pickle
    np.testing.assert_array_equal(r2.read("u10", region), before)


def test_chunk_shape_is_decode_unit(multi_stores, zarr3_sharded, netcdf_file):
    assert open_cube(zarr3_sharded).chunk_shape("sst") == (6, 20, 20)
    assert open_cube(netcdf_file).chunk_shape("sst") == (6, 20, 20)
    paths, _ = multi_stores
    reader = MultiCubeReader(paths, dim="store")
    assert reader.chunk_shape("u10") == reader.block_shape("u10")
