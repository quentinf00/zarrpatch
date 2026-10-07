import numpy as np
import pytest
import xarray as xr

DIMS = {"time": 24, "lat": 60, "lon": 80}


def make_dataset() -> xr.Dataset:
    rng = np.random.default_rng(42)
    shape = tuple(DIMS.values())
    return xr.Dataset(
        {
            "sst": (list(DIMS), rng.normal(size=shape).astype("float32")),
            "ssh": (list(DIMS), rng.normal(size=shape).astype("float32")),
            "mask": (["lat", "lon"], rng.integers(0, 2, size=shape[1:]).astype("int8")),
        },
        coords={
            "time": np.arange(DIMS["time"]),
            "lat": np.linspace(-30, 30, DIMS["lat"]),
            "lon": np.linspace(0, 80, DIMS["lon"]),
        },
    )


@pytest.fixture(scope="session")
def dataset() -> xr.Dataset:
    return make_dataset()


CHUNKS = {"time": 6, "lat": 20, "lon": 20}
SHARDS = {"time": 12, "lat": 40, "lon": 40}


def _var_encoding(ds, per_var):
    return {
        v: {k: tuple(spec[d] for d in ds[v].dims if d in spec) or None
            for k, spec in per_var.items()}
        for v in ds.data_vars
    }


@pytest.fixture(scope="session")
def zarr3_sharded(dataset, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "cube_v3_sharded.zarr"
    enc = _var_encoding(dataset, {"chunks": CHUNKS, "shards": SHARDS})
    dataset.to_zarr(path, zarr_format=3, encoding=enc, mode="w")
    return str(path)


@pytest.fixture(scope="session")
def zarr3_plain(dataset, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "cube_v3.zarr"
    enc = _var_encoding(dataset, {"chunks": CHUNKS})
    dataset.to_zarr(path, zarr_format=3, encoding=enc, mode="w")
    return str(path)


@pytest.fixture(scope="session")
def zarr2_plain(dataset, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "cube_v2.zarr"
    enc = _var_encoding(dataset, {"chunks": CHUNKS})
    dataset.to_zarr(path, zarr_format=2, encoding=enc, mode="w")
    return str(path)


@pytest.fixture(scope="session")
def netcdf_file(dataset, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "cube.nc"
    enc = {
        v: {"chunksizes": tuple(CHUNKS[d] for d in dataset[v].dims if d in CHUNKS)}
        for v in dataset.data_vars
    }
    dataset.to_netcdf(path, encoding=enc)
    return str(path)


@pytest.fixture(scope="session")
def zarr_zip(dataset, tmp_path_factory):
    """A ``.zarr.zip`` store: write to a v3 zarr dir, then zip it into a store.

    The store metadata/chunks sit at the archive root (no wrapping directory),
    which is what :class:`zarr.storage.ZipStore` expects.
    """
    import zipfile

    src = tmp_path_factory.mktemp("data") / "cube_for_zip.zarr"
    enc = _var_encoding(dataset, {"chunks": CHUNKS})
    dataset.to_zarr(src, zarr_format=3, encoding=enc, mode="w")

    path = tmp_path_factory.mktemp("data") / "cube.zarr.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src).as_posix())
    return str(path)


@pytest.fixture(scope="session")
def multi_stores(tmp_path_factory):
    """Three same-schema stores, one with a ragged leading ``member`` dim.

    Returns ``(paths, arrays)`` where ``arrays[i]`` is the ``(member, lat, lon)``
    field of store ``i`` (member length varies: 1, 3, 2) for direct comparison.
    Chunked ``(1, 20, 20)`` so ``block_shape`` is well defined.
    """
    d = tmp_path_factory.mktemp("multi")
    sizes = [1, 3, 2]
    paths, arrays = [], []
    for i, m in enumerate(sizes):
        rng = np.random.default_rng(100 + i)
        field = rng.normal(size=(m, DIMS["lat"], DIMS["lon"])).astype("float32")
        ds = xr.Dataset(
            {"u10": (["member", "lat", "lon"], field),
             "v10": (["member", "lat", "lon"], field * 2)},
            coords={"member": np.arange(m),
                    "lat": np.linspace(-30, 30, DIMS["lat"]),
                    "lon": np.linspace(0, 80, DIMS["lon"])},
        )
        enc = {v: {"chunks": (1, 20, 20)} for v in ds.data_vars}
        path = str(d / f"store_{i}.zarr")
        ds.to_zarr(path, zarr_format=3, encoding=enc, mode="w")
        paths.append(path)
        arrays.append(field)
    return paths, arrays


@pytest.fixture(scope="session")
def kerchunk_refs(netcdf_file, tmp_path_factory):
    import json

    from kerchunk.hdf import SingleHdf5ToZarr

    path = tmp_path_factory.mktemp("data") / "cube_kerchunk.json"
    refs = SingleHdf5ToZarr(netcdf_file).translate()
    path.write_text(json.dumps(refs))
    return str(path)


def pytest_collection_modifyitems(config, items):
    """Run TensorStore tests last.

    TensorStore aborts any process forked after it has started its threads, so
    once a test reads through it in the pytest process, later tests that fork
    DataLoader workers would abort. Moving these tests to the end keeps the
    rest of the suite independent of file ordering.
    """
    items.sort(key=lambda item: "tensorstore" in item.nodeid)
