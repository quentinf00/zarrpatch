"""TensorStoreReader: same metadata and same patches as the zarr backend.

conftest.py runs this module last (TensorStore forbids fork after use).
"""

import pickle
import warnings

import numpy as np
import pytest
import zarr
from torch.utils.data import DataLoader

from zarrpatch import (
    ListSampler,
    MultiCubeReader,
    RandomSampler,
    TensorStoreReader,
    ZarrPatchDataset,
    ZarrReader,
    open_cube,
)

pytest.importorskip("tensorstore")

PATCH = {"time": 4, "lat": 16, "lon": 16}
CORNERS = [
    {"time": 0, "lat": 0, "lon": 0},
    {"time": 20, "lat": 44, "lon": 64},
    {"time": 7, "lat": 13, "lon": 33},
    {"time": 5, "lat": 19, "lon": 39},  # straddles chunk edges on every dim
]


@pytest.fixture(autouse=True)
def _quiet_fork_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TensorStoreReader is reading")
        yield


@pytest.fixture(params=["zarr3_sharded", "zarr3_plain", "zarr2_plain", "zarr_zip"])
def zarr_path(request):
    return request.getfixturevalue(request.param)


def test_open_cube_backend(zarr3_plain):
    reader = open_cube(zarr3_plain, backend="tensorstore", threads=2)
    assert isinstance(reader, TensorStoreReader) and reader.threads == 2
    with pytest.raises(ValueError, match="threads"):
        TensorStoreReader(zarr3_plain, threads=0)


def test_metadata_matches_zarr_backend(zarr_path):
    ts_reader, ref = TensorStoreReader(zarr_path), ZarrReader(zarr_path)
    assert ts_reader.variables == ref.variables
    for v in ref.variables:
        assert ts_reader.dims(v) == ref.dims(v)
        assert ts_reader.shape(v) == ref.shape(v)
        assert ts_reader.dtype(v) == ref.dtype(v)
        assert ts_reader.block_shape(v) == ref.block_shape(v)
        assert ts_reader.chunk_shape(v) == ref.chunk_shape(v)
        ts_reader._ts_array(v)  # the cross-check against TensorStore passes


def test_patches_match_direct_slicing_and_zarr_backend(zarr_path, dataset):
    kwargs = dict(variables=["sst", "ssh", "mask"], sampler=ListSampler(CORNERS),
                  shuffle=False, return_corners=True)
    got = {tuple(p["corner"]): p for p in ZarrPatchDataset(
        TensorStoreReader(zarr_path), PATCH, **kwargs)}
    ref = {tuple(p["corner"]): p for p in ZarrPatchDataset(zarr_path, PATCH, **kwargs)}
    assert got.keys() == ref.keys() and len(got) == len(CORNERS)
    for c in CORNERS:
        key = (c["lat"], c["lon"], c["time"])  # sorted(PATCH) order
        t, la, lo = (slice(c[d], c[d] + PATCH[d]) for d in ("time", "lat", "lon"))
        for var in ("sst", "ssh"):
            np.testing.assert_array_equal(got[key][var], dataset[var].values[t, la, lo])
        np.testing.assert_array_equal(got[key]["mask"], dataset["mask"].values[la, lo])
        for var in ("sst", "ssh", "mask"):
            assert got[key][var].dtype == ref[key][var].dtype
            np.testing.assert_array_equal(got[key][var], ref[key][var])


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("dtype, fill", [("float32", np.nan), ("int16", -999)])
def test_fill_values_match_zarr_backend(tmp_path, zarr_format, dtype, fill):
    path = str(tmp_path / "sparse.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    kw = ({"dimension_names": ("y", "x")} if zarr_format == 3 else {})
    arr = root.create_array("v", shape=(40, 40), chunks=(10, 10), dtype=dtype,
                            fill_value=fill, **kw)
    if zarr_format == 2:
        arr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    arr[:10, :20] = np.arange(200, dtype=dtype).reshape(10, 20)  # other chunks absent
    region = {"y": slice(5, 25), "x": slice(15, 35)}
    got = TensorStoreReader(path).read("v", region)
    ref = ZarrReader(path).read("v", region)
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)  # NaN positions compared equal too


def test_read_many_matches_read(zarr3_sharded):
    reader = TensorStoreReader(zarr3_sharded)
    region = {"time": slice(3, 9), "lat": slice(10, 50), "lon": slice(0, 30)}
    many = reader.read_many(["sst", "ssh", "mask"], region)
    for v in ("sst", "ssh", "mask"):
        np.testing.assert_array_equal(many[v], reader.read(v, region))


def test_disagreement_with_zarr_metadata_fails_loudly(zarr3_plain, monkeypatch):
    reader = TensorStoreReader(zarr3_plain)
    monkeypatch.setattr(reader, "chunk_shape", lambda v: (1, 1, 1))
    with pytest.raises(ValueError, match="disagree"):
        reader.read("sst", {})


def test_unopenable_array_fails_loudly(zarr3_plain, tmp_path):
    import shutil

    broken = tmp_path / "broken.zarr"
    shutil.copytree(zarr3_plain, broken)
    reader = TensorStoreReader(str(broken))
    reader.variables  # zarr-python metadata is loaded before the corruption
    (broken / "sst" / "zarr.json").write_text("{not json")
    with pytest.raises(ValueError, match="backend='zarr'"):
        reader.read("sst", {"time": slice(0, 1)})


def test_pickling_drops_handles(zarr3_sharded):
    reader = TensorStoreReader(zarr3_sharded)
    reader.read("sst", {"time": slice(0, 1)})
    assert reader._ts
    clone = pickle.loads(pickle.dumps(reader))
    assert clone._ts == {} and clone._group is None
    np.testing.assert_array_equal(clone.read("sst", {"time": slice(0, 2)}),
                                  reader.read("sst", {"time": slice(0, 2)}))


def test_warns_once_about_fork_in_main_process(zarr3_plain, monkeypatch):
    monkeypatch.setattr(TensorStoreReader, "_fork_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TensorStoreReader(zarr3_plain).read("sst", {"time": slice(0, 1)})
        TensorStoreReader(zarr3_plain).read("sst", {"time": slice(0, 1)})
    assert sum("forkserver" in str(w.message) for w in caught) == 1


def test_multiworker_forkserver_yields_each_patch_once(zarr3_sharded):
    ds = ZarrPatchDataset(
        TensorStoreReader(zarr3_sharded), PATCH, sampler=RandomSampler(120),
        return_corners=True, variables=["sst"],
    )
    ref = sorted(tuple(p["corner"]) for p in ZarrPatchDataset(
        zarr3_sharded, PATCH, sampler=RandomSampler(120), return_corners=True,
        variables=["sst"]))
    loader = DataLoader(ds, batch_size=16, num_workers=2,
                        multiprocessing_context="forkserver")
    got = sorted(tuple(c.tolist()) for b in loader for c in b["corner"])
    assert got == ref


def test_multicube_reader_with_tensorstore_backend(multi_stores):
    paths, arrays = multi_stores
    reader = MultiCubeReader(paths, dim="store", backend="tensorstore",
                             variables=["u10"])
    # member dim is ragged, so read only member 0 of each store
    reader_units = MultiCubeReader([(p, {"member": 0}) for p in paths],
                                   dim="store", backend="tensorstore")
    region = {"store": slice(0, 3), "lat": slice(5, 30), "lon": slice(10, 40)}
    got = reader_units.read("u10", region)
    for i, field in enumerate(arrays):
        np.testing.assert_array_equal(got[i], field[0, 5:30, 10:40])
    assert isinstance(reader._reader(paths[0]), TensorStoreReader)
