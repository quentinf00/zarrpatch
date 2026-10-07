"""Backends for reading regions of a datacube.

All readers expose the same minimal interface:

- ``variables``: list of variable names
- ``dims(var)``: dimension names of a variable
- ``shape(var)``: shape of a variable
- ``block_shape(var)``: the natural I/O unit of the storage — the shard shape
  for sharded zarr v3 arrays, the chunk shape otherwise. Patch requests are
  grouped by this unit so each block is read (and decompressed) once.
- ``chunk_shape(var)``: the decode unit — the (inner) chunk shape. A read
  decompresses every chunk its region touches, so this sets decoded bytes.
- ``dtype(var)``: numpy dtype of a variable.
- ``read(var, region)``: read a region (dict of dim -> slice) as numpy.
- ``read_many(vars, region)``: the same region of several variables, as a
  dict; backends with async I/O issue the reads concurrently.

Readers are pickled into dataloader worker processes, so they open their file
handles lazily and drop them when pickled.
"""

from __future__ import annotations

import os
import warnings
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .profiling import JsonlProfiler, emit

Region = Dict[str, slice]

# A source cube unit: a path, or a (path, {dim: index}) pair that squeezes the
# selected integer indices out of the cube (for ragged/reducible dims).
Source = Union[str, Tuple[str, Dict[str, int]]]


class CubeReader:
    """Base class, see module docstring for the interface."""

    profiler: Optional[JsonlProfiler] = None

    def set_profiler(self, profiler: Optional[JsonlProfiler]) -> None:
        """Attach an optional profiler, including after reader construction."""
        self.profiler = profiler

    @property
    def variables(self) -> List[str]:
        raise NotImplementedError

    def dims(self, var: str) -> Tuple[str, ...]:
        raise NotImplementedError

    def shape(self, var: str) -> Tuple[int, ...]:
        raise NotImplementedError

    def block_shape(self, var: str) -> Tuple[int, ...]:
        raise NotImplementedError

    def chunk_shape(self, var: str) -> Tuple[int, ...]:
        """Decode unit; defaults to the block shape (no sharding)."""
        return self.block_shape(var)

    def dtype(self, var: str) -> np.dtype:
        raise NotImplementedError

    def read(self, var: str, region: Region) -> np.ndarray:
        raise NotImplementedError

    def read_many(self, variables: Sequence[str],
                  region: Region) -> Dict[str, np.ndarray]:
        """Read ``region`` of several variables (sequentially by default)."""
        return {v: self.read(v, region) for v in variables}

    # -- helpers shared by all backends -------------------------------------

    def dim_sizes(self, var: str) -> Dict[str, int]:
        return dict(zip(self.dims(var), self.shape(var)))

    def _slices(self, var: str, region: Region) -> Tuple[slice, ...]:
        """Region dict -> positional slice tuple (full extent for absent dims)."""
        return tuple(region.get(d, slice(None)) for d in self.dims(var))

    def __getstate__(self):
        state = self.__dict__.copy()
        state[self._handle_attr] = None
        return state


class ZarrReader(CubeReader):
    """Reads a zarr group (v2 or v3, sharded or not) written e.g. by xarray.

    A ``.zarr.zip`` (any ``.zip``) path is opened through
    :class:`zarr.storage.ZipStore` in read-only mode. Zip handles are
    per-process: :meth:`__getstate__` drops the open store when the reader is
    pickled into a dataloader worker, and each worker reopens it lazily on its
    first ``.group`` access.
    """

    _handle_attr = "_group"

    def __init__(self, path: str, profiler: Optional[JsonlProfiler] = None):
        self.path = str(path)
        self._group = None
        self.profiler = profiler

    @property
    def group(self):
        if self._group is None:
            import zarr

            emit(self.profiler, "store_open_start", path=self.path, backend="zarr")
            start = time.monotonic_ns()
            try:
                if self.path.endswith(".zip"):
                    from zarr.storage import ZipStore

                    store = ZipStore(self.path, mode="r")
                    self._group = zarr.open_group(store, mode="r")
                else:
                    self._group = zarr.open_group(self.path, mode="r")
            except BaseException as exc:
                emit(self.profiler, "store_open_error", path=self.path,
                     backend="zarr",
                     elapsed_ns=time.monotonic_ns() - start,
                     error_type=type(exc).__name__)
                raise
            emit(self.profiler, "store_open_end", path=self.path, backend="zarr",
                 elapsed_ns=time.monotonic_ns() - start)
        return self._group

    @property
    def variables(self) -> List[str]:
        # exclude coordinate arrays (1D arrays named after their dimension,
        # as written by xarray), like the netCDF backend does
        return sorted(
            name for name in self.group.array_keys()
            if self.dims(name) != (name,)
        )

    def _array(self, var: str):
        return self.group[var]

    def dims(self, var: str) -> Tuple[str, ...]:
        arr = self._array(var)
        names = getattr(arr.metadata, "dimension_names", None)  # zarr v3
        if names is None:
            names = arr.attrs.get("_ARRAY_DIMENSIONS")  # zarr v2 xarray convention
        if names is None:
            names = [f"dim_{i}" for i in range(arr.ndim)]
        return tuple(names)

    def shape(self, var: str) -> Tuple[int, ...]:
        return tuple(self._array(var).shape)

    def block_shape(self, var: str) -> Tuple[int, ...]:
        arr = self._array(var)
        shards = getattr(arr, "shards", None)
        return tuple(shards) if shards is not None else tuple(arr.chunks)

    def chunk_shape(self, var: str) -> Tuple[int, ...]:
        return tuple(self._array(var).chunks)

    def dtype(self, var: str) -> np.dtype:
        return np.dtype(self._array(var).dtype)

    def read(self, var: str, region: Region) -> np.ndarray:
        start = time.monotonic_ns()
        emit(self.profiler, "variable_read_start", detail=True,
             path=self.path, variable=var)
        out = np.asarray(self._array(var)[self._slices(var, region)])
        emit(self.profiler, "variable_read_end", detail=True, path=self.path,
             variable=var, bytes=out.nbytes,
             elapsed_ns=time.monotonic_ns() - start)
        return out


class KerchunkReader(ZarrReader):
    """Reads a kerchunk/virtualizarr reference file (JSON or parquet).

    The references describe a virtual zarr store over one or more source files
    (e.g. netCDF/HDF5 archives), read through fsspec's ReferenceFileSystem.
    ``storage_options`` are passed to the *target* filesystem holding the
    source files (e.g. ``{"remote_protocol": "s3", "remote_options": ...}``).
    """

    def __init__(self, path: str, storage_options: Optional[dict] = None,
                 profiler: Optional[JsonlProfiler] = None):
        super().__init__(path, profiler=profiler)
        self.storage_options = dict(storage_options or {})

    @property
    def group(self):
        if self._group is None:
            import fsspec
            import zarr
            from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
            from zarr.storage import FsspecStore

            emit(self.profiler, "store_open_start", path=self.path,
                 backend="kerchunk")
            start = time.monotonic_ns()
            try:
                fs = fsspec.filesystem("reference", fo=self.path,
                                       **self.storage_options)
                store = FsspecStore(
                    AsyncFileSystemWrapper(fs, asynchronous=True),
                    read_only=True,
                )
                self._group = zarr.open_group(store, mode="r")
            except BaseException as exc:
                emit(self.profiler, "store_open_error", path=self.path,
                     backend="kerchunk", elapsed_ns=time.monotonic_ns() - start,
                     error_type=type(exc).__name__)
                raise
            emit(self.profiler, "store_open_end", path=self.path,
                 backend="kerchunk", elapsed_ns=time.monotonic_ns() - start)
        return self._group


class TensorStoreReader(ZarrReader):
    """Reads a zarr store (v2 or v3, sharded or not) through TensorStore.

    Metadata — variables, dimension names, shapes, dtypes, shard and chunk
    shapes — comes from zarr-python exactly as for :class:`ZarrReader`, so
    both backends describe a store identically. Region reads go through
    TensorStore's C++ engine, which decodes a grouped read with much less
    per-chunk overhead than zarr-python; ``read_many`` issues all variables'
    reads concurrently. Each array is cross-checked (shape, dtype, shard and
    chunk layout) the first time it is read, and anything TensorStore cannot
    open raises ``ValueError`` suggesting ``backend="zarr"``.

    Parameters
    ----------
    path:
        Local path, ``.zip`` store, or a URL TensorStore understands
        (``s3://``, ``gs://``, ``http(s)://``; remote use is untested).
    threads:
        TensorStore decode concurrency (``data_copy_concurrency``) per
        process. The default of 1 suits one reader per DataLoader worker;
        raise it for single-process loading.

    No chunk cache is configured: block-grouped reads already decode each
    chunk once per group, and a cache smaller than the cube does not help
    random access (see ``benchmarks/results_ecosystem.md``).

    Fork safety: TensorStore aborts a process forked after it has started its
    threads. Handles are therefore opened lazily in the reading process, never
    at construction, and dropped on pickling. Reading in the main process and
    then starting a ``fork`` DataLoader aborts the workers — use
    ``multiprocessing_context="forkserver"`` (or ``"spawn"``), as recommended
    for zarrpatch anyway; a one-time warning flags the risky case.
    """

    _fork_warned = False

    def __init__(self, path: str, threads: int = 1,
                 profiler: Optional[JsonlProfiler] = None):
        super().__init__(path, profiler=profiler)
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        self.threads = threads
        self._ts: Dict[str, object] = {}
        self._ts_pid: Optional[int] = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_ts"] = {}
        state["_ts_pid"] = None
        return state

    def _kvstore(self, var: str):
        if "://" in self.path:
            return f"{self.path.rstrip('/')}/{var}/"
        if self.path.endswith(".zip"):
            return {"driver": "zip",
                    "base": {"driver": "file",
                             "path": os.path.abspath(self.path)},
                    "path": f"{var}/"}
        return {"driver": "file",
                "path": os.path.join(os.path.abspath(self.path), var, "")}

    def _warn_fork_risk(self) -> None:
        import multiprocessing as mp

        if mp.parent_process() is None and not TensorStoreReader._fork_warned:
            TensorStoreReader._fork_warned = True
            warnings.warn(
                "TensorStoreReader is reading in the main process: a DataLoader "
                "started afterwards with the default 'fork' start method will "
                "abort its workers (TensorStore forbids fork after use). Use "
                "multiprocessing_context='forkserver' or 'spawn'."
            )

    def _ts_array(self, var: str):
        if self._ts_pid != os.getpid():  # new process: never reuse handles
            self._ts, self._ts_pid = {}, os.getpid()
        if var in self._ts:
            return self._ts[var]
        import tensorstore as ts

        self._warn_fork_risk()
        fmt = getattr(self._array(var).metadata, "zarr_format", 3)
        context = ts.Context({
            "cache_pool": {"total_bytes_limit": 0},
            "data_copy_concurrency": {"limit": self.threads},
        })
        emit(self.profiler, "array_open_start", path=self.path, variable=var,
             backend="tensorstore")
        start = time.monotonic_ns()
        try:
            arr = ts.open({"driver": "zarr3" if fmt == 3 else "zarr",
                           "kvstore": self._kvstore(var)},
                          read=True, context=context).result()
        except Exception as exc:
            emit(self.profiler, "array_open_error", path=self.path,
                 variable=var, backend="tensorstore",
                 elapsed_ns=time.monotonic_ns() - start,
                 error_type=type(exc).__name__)
            raise ValueError(
                f"TensorStore cannot open {self.path!r} variable {var!r} "
                f"({exc}); read this store with backend='zarr'"
            ) from exc
        emit(self.profiler, "array_open_end", path=self.path, variable=var,
             backend="tensorstore", elapsed_ns=time.monotonic_ns() - start)
        layout = arr.chunk_layout
        seen = {
            "shape": tuple(arr.shape),
            "dtype": np.dtype(arr.dtype.numpy_dtype),
            "block_shape": tuple(layout.write_chunk.shape),
            "chunk_shape": tuple(layout.read_chunk.shape),
        }
        expected = {
            "shape": self.shape(var), "dtype": self.dtype(var),
            "block_shape": self.block_shape(var),
            "chunk_shape": self.chunk_shape(var),
        }
        diff = {k: (seen[k], expected[k]) for k in seen if seen[k] != expected[k]}
        if diff:
            raise ValueError(
                f"TensorStore and zarr-python disagree on {self.path!r} "
                f"variable {var!r} (tensorstore, zarr): {diff}; read this "
                "store with backend='zarr'"
            )
        self._ts[var] = arr
        return arr

    def read(self, var: str, region: Region) -> np.ndarray:
        return self.read_many([var], region)[var]

    def read_many(self, variables: Sequence[str],
                  region: Region) -> Dict[str, np.ndarray]:
        start = time.monotonic_ns()
        futures = {}
        for v in variables:
            emit(self.profiler, "variable_read_start", detail=True,
                 path=self.path, variable=v)
            futures[v] = self._ts_array(v)[self._slices(v, region)].read()
        out = {}
        for v, future in futures.items():
            out[v] = np.asarray(future.result())
            emit(self.profiler, "variable_read_end", detail=True,
                 path=self.path, variable=v, bytes=out[v].nbytes,
                 elapsed_ns=time.monotonic_ns() - start)
        return out


class NetCDFReader(CubeReader):
    """Reads a netCDF4 file. Provided mainly as a benchmark baseline."""

    _handle_attr = "_ds"

    def __init__(self, path: str, profiler: Optional[JsonlProfiler] = None):
        self.path = str(path)
        self._ds = None
        self.profiler = profiler

    @property
    def ds(self):
        if self._ds is None:
            import netCDF4

            emit(self.profiler, "store_open_start", path=self.path,
                 backend="netcdf")
            start = time.monotonic_ns()
            try:
                self._ds = netCDF4.Dataset(self.path, mode="r")
            except BaseException as exc:
                emit(self.profiler, "store_open_error", path=self.path,
                     backend="netcdf", elapsed_ns=time.monotonic_ns() - start,
                     error_type=type(exc).__name__)
                raise
            emit(self.profiler, "store_open_end", path=self.path,
                 backend="netcdf", elapsed_ns=time.monotonic_ns() - start)
        return self._ds

    @property
    def variables(self) -> List[str]:
        # exclude coordinate variables (1D vars named after their dimension)
        return sorted(
            name
            for name, v in self.ds.variables.items()
            if not (v.ndim == 1 and v.dimensions == (name,))
        )

    def dims(self, var: str) -> Tuple[str, ...]:
        return tuple(self.ds.variables[var].dimensions)

    def shape(self, var: str) -> Tuple[int, ...]:
        return tuple(self.ds.variables[var].shape)

    def block_shape(self, var: str) -> Tuple[int, ...]:
        chunking = self.ds.variables[var].chunking()
        if chunking is None or chunking == "contiguous":
            return self.shape(var)
        return tuple(chunking)

    def dtype(self, var: str) -> np.dtype:
        return np.dtype(self.ds.variables[var].dtype)

    def read(self, var: str, region: Region) -> np.ndarray:
        start = time.monotonic_ns()
        emit(self.profiler, "variable_read_start", detail=True,
             path=self.path, variable=var)
        out = np.asarray(self.ds.variables[var][self._slices(var, region)])
        emit(self.profiler, "variable_read_end", detail=True, path=self.path,
             variable=var, bytes=out.nbytes,
             elapsed_ns=time.monotonic_ns() - start)
        return out


def open_cube(path: str, backend: Optional[str] = None, **kwargs) -> CubeReader:
    """Open a datacube, guessing the backend from the file extension.

    ``.nc``/``.nc4``/``.cdf`` → netCDF, ``.json`` → kerchunk, everything else
    (including ``.zarr`` and ``.zarr.zip``/``.zip``) → the zarr backend.
    ``backend="tensorstore"`` reads a zarr store through
    :class:`TensorStoreReader` (optional ``tensorstore`` dependency).
    """
    if backend is None:
        p = str(path)
        if p.endswith((".nc", ".nc4", ".cdf")):
            backend = "netcdf"
        elif p.endswith(".json"):
            backend = "kerchunk"
        else:
            backend = "zarr"
    if backend == "zarr":
        return ZarrReader(path, **kwargs)
    if backend == "netcdf":
        return NetCDFReader(path, **kwargs)
    if backend == "kerchunk":
        return KerchunkReader(path, **kwargs)
    if backend == "tensorstore":
        return TensorStoreReader(path, **kwargs)
    raise ValueError(f"unknown backend {backend!r}, expected 'zarr', "
                     f"'tensorstore', 'netcdf' or 'kerchunk'")


def expand_units(paths: Sequence[str], dim: str,
                 backend: Optional[str] = None) -> List[Tuple[str, Dict[str, int]]]:
    """Expand stores into one ``(path, {dim: i})`` unit per index along ``dim``.

    Reads each store's length along ``dim`` lazily, opening every store exactly
    once. Use for :class:`MultiCubeReader` when the stores share every dim
    except a ragged one (e.g. per-event stores with different ``init_time`` lengths):
    ``expand_units(paths, "init_time")`` yields uniform per-unit sub-cubes.
    """
    units: List[Tuple[str, Dict[str, int]]] = []
    for path in paths:
        reader = open_cube(path, backend)
        n = None
        for v in reader.variables:
            sizes = reader.dim_sizes(v)
            if dim in sizes:
                n = sizes[dim]
                break
        if n is None:
            raise ValueError(f"dim {dim!r} not found in any variable of {path!r}")
        units.extend((path, {dim: i}) for i in range(n))
    return units


class MultiCubeReader(CubeReader):
    """Concatenate N same-schema cube units along a new virtual leading dim.

    Each *unit* is a fixed-shape sub-cube given as ``(path, select)`` where
    ``select`` maps dim -> integer index; the selected dims are squeezed out
    (so ragged/reducible dims can be collapsed to make the remaining shapes
    uniform — see :func:`expand_units`). A bare ``path`` is shorthand for an
    empty ``select`` (whole-store concatenation of uniform stores). Units are
    presented as one cube with a new leading dim ``dim`` (default ``"store"``)
    of length ``n_units`` and **block size 1** along it, so block groups never
    cross units and each patch reads exactly one unit.

    Per variable, the exposed schema is ``dims = (dim,) + base_dims``,
    ``shape = (n_units,) + base_shape``, ``block_shape = (1,) + base_block``,
    where the base schema comes from unit 0 after its ``select`` squeeze.

    Stores are opened lazily and cached in an LRU of at most ``cache_size``
    open readers keyed by path (many units share a path). Unit 0 is opened at
    construction to fix the reference schema; every other unit is validated on
    first read of its path — a missing variable or a mismatched
    dims/shape/dtype raises ``ValueError``; a mismatched ``block_shape`` warns
    once (reads still work, but that unit may re-decode a chunk).

    Requires ``patch_size[dim] == 1`` in the consuming dataset (checked via
    :meth:`check_patch_size`); everything else in the patch/block/sampler
    pipeline works unchanged, and ``WeightedSampler(dims=[dim])`` gives
    per-unit oversampling for free.
    """

    _handle_attr = "_cache"

    def __init__(self, sources: Sequence[Source], dim: str = "store",
                 backend: Optional[str] = None, cache_size: int = 8,
                 variables: Optional[Sequence[str]] = None,
                 profiler: Optional[JsonlProfiler] = None):
        if len(sources) == 0:
            raise ValueError("MultiCubeReader needs at least one source")
        self.dim = dim
        self._backend = backend
        self._cache_size = cache_size
        self._sources: List[Tuple[str, Dict[str, int]]] = [
            (s, {}) if isinstance(s, str) else (s[0], dict(s[1]))
            for s in sources
        ]
        self._n_units = len(self._sources)
        self._cache: Optional[OrderedDict] = None
        self._validated: set = set()
        self._block_warned = False
        self.profiler = profiler

        # reference schema from unit 0, after its select squeeze
        path0, select0 = self._sources[0]
        reader0 = self._reader(path0)
        available0 = list(reader0.variables)
        if variables is None:
            self._ref_variables = available0
        else:
            missing = set(variables) - set(available0)
            if missing:
                raise ValueError(
                    f"unit {path0!r} is missing requested variables "
                    f"{sorted(missing)}"
                )
            self._ref_variables = list(variables)
        self._base_dims: Dict[str, Tuple[str, ...]] = {}
        self._base_shape: Dict[str, Tuple[int, ...]] = {}
        self._base_block: Dict[str, Tuple[int, ...]] = {}
        self._base_chunk: Dict[str, Tuple[int, ...]] = {}
        self._dtype: Dict[str, np.dtype] = {}
        for v in self._ref_variables:
            full_dims = reader0.dims(v)
            if dim in full_dims:
                raise ValueError(
                    f"virtual dim {dim!r} already exists in variable {v!r}"
                )
            keep = [i for i, d in enumerate(full_dims) if d not in select0]
            shp, blk = reader0.shape(v), reader0.block_shape(v)
            self._base_dims[v] = tuple(full_dims[i] for i in keep)
            self._base_shape[v] = tuple(shp[i] for i in keep)
            self._base_block[v] = tuple(blk[i] for i in keep)
            chk = reader0.chunk_shape(v)
            self._base_chunk[v] = tuple(chk[i] for i in keep)
            self._dtype[v] = np.dtype(reader0.dtype(v))
        self._validated.add(path0)

    # -- LRU of open per-path readers ---------------------------------------

    def _reader(self, path: str) -> CubeReader:
        if self._cache is None:
            self._cache = OrderedDict()
        cache = self._cache
        if path in cache:
            emit(self.profiler, "reader_cache_hit", detail=True, path=path)
            cache.move_to_end(path)
            return cache[path]
        emit(self.profiler, "reader_cache_miss", path=path)
        reader = open_cube(path, self._backend, profiler=self.profiler)
        cache[path] = reader
        while len(cache) > self._cache_size:
            evicted, _ = cache.popitem(last=False)
            emit(self.profiler, "reader_cache_evict", path=evicted)
        return reader

    def set_profiler(self, profiler: Optional[JsonlProfiler]) -> None:
        self.profiler = profiler
        if self._cache is not None:
            for reader in self._cache.values():
                reader.set_profiler(profiler)

    def _validate(self, path: str, select: Dict[str, int]) -> None:
        if path in self._validated:
            return
        reader = self._reader(path)
        available = set(reader.variables)
        for v in self._ref_variables:
            if v not in available:
                raise ValueError(
                    f"unit {path!r} is missing variable {v!r}"
                )
            full_dims = reader.dims(v)
            keep = [i for i, d in enumerate(full_dims) if d not in select]
            dims = tuple(full_dims[i] for i in keep)
            if dims != self._base_dims[v]:
                raise ValueError(
                    f"unit {path!r} variable {v!r} has dims {dims} != "
                    f"reference {self._base_dims[v]}"
                )
            shp = reader.shape(v)
            shape = tuple(shp[i] for i in keep)
            if shape != self._base_shape[v]:
                raise ValueError(
                    f"unit {path!r} variable {v!r} has shape {shape} != "
                    f"reference {self._base_shape[v]}"
                )
            if np.dtype(reader.dtype(v)) != self._dtype[v]:
                raise ValueError(
                    f"unit {path!r} variable {v!r} has dtype "
                    f"{reader.dtype(v)} != reference {self._dtype[v]}"
                )
            blk = reader.block_shape(v)
            block = tuple(blk[i] for i in keep)
            if block != self._base_block[v] and not self._block_warned:
                warnings.warn(
                    f"unit {path!r} variable {v!r} has block_shape {block} != "
                    f"reference {self._base_block[v]}; grouping uses unit 0's "
                    "layout, so this unit may re-decode a chunk"
                )
                self._block_warned = True
        self._validated.add(path)

    # -- schema exposure ----------------------------------------------------

    @property
    def variables(self) -> List[str]:
        return list(self._ref_variables)

    def dims(self, var: str) -> Tuple[str, ...]:
        return (self.dim,) + self._base_dims[var]

    def shape(self, var: str) -> Tuple[int, ...]:
        return (self._n_units,) + self._base_shape[var]

    def block_shape(self, var: str) -> Tuple[int, ...]:
        return (1,) + self._base_block[var]

    def chunk_shape(self, var: str) -> Tuple[int, ...]:
        return (1,) + self._base_chunk[var]

    def dtype(self, var: str) -> np.dtype:
        return self._dtype[var]

    def check_patch_size(self, patch_size: Dict[str, int]) -> None:
        """Require the virtual dim to be a size-1 patch dim.

        Called by :class:`~zarrpatch.dataset.ZarrPatchDataset` at build time so
        block groups stay within one unit.
        """
        if self.dim not in patch_size:
            raise ValueError(
                f"patch_size must include the virtual dim {self.dim!r}"
            )
        if patch_size[self.dim] != 1:
            raise ValueError(
                f"patch_size[{self.dim!r}] must be 1 (the virtual dim is a "
                f"size-1 patch dim), got {patch_size[self.dim]}"
            )

    def read(self, var: str, region: Region) -> np.ndarray:
        region = dict(region)
        sl = region.pop(self.dim, slice(None))
        units = range(*sl.indices(self._n_units))
        out = []
        for u in units:
            path, select = self._sources[u]
            self._validate(path, select)
            reader = self._reader(path)
            sub = {**region, **{d: slice(i, i + 1) for d, i in select.items()}}
            a = reader.read(var, sub)
            full_dims = reader.dims(var)
            keep = [ax for ax, d in enumerate(full_dims) if d not in select]
            a = a.reshape(tuple(a.shape[ax] for ax in keep))
            out.append(a)
        return np.stack(out, axis=0)
