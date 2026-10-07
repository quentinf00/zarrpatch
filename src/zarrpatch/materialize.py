"""Freeze a :class:`~zarrpatch.dataset.ZarrPatchDataset` to an HDF5 snapshot.

The live dataset stores nothing extra and draws fresh corners every epoch. When
the corner set is fixed (validation, or a once-drawn weighted training pool) that
flexibility is not needed, and reading assembled patches back from a flat file is
dramatically faster than re-reading and re-assembling the cube. :func:`materialize`
does one sequential pass over the dataset — exactly the access pattern the
block-grouped reader is best at — and writes one HDF5 dataset per variable, one
sample per chunk, plus a ``corner`` dataset. :class:`HDF5PatchDataset` reads that
file back as a map-style ``torch.utils.data.Dataset`` that composes with stock
torch samplers (``DistributedSampler`` for DDP, ``WeightedRandomSampler`` over
indices for a reweighted frozen pool).

Whatever the dataset is configured to produce is what gets frozen: attach a
``transform`` to the reader for a raw (pre-transform) snapshot with augmentation
still live, or to the assembled sample for a post-transform snapshot that also
skips the assemble cost. The choice is where you attach the callable; there is no
mode flag.

``h5py`` is an optional dependency (``pip install zarrpatch[hdf5]``); it is
imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from itertools import chain
from math import prod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from .dataset import Patch, ZarrPatchDataset

_LOG = logging.getLogger("zarrpatch")
MATERIALIZED_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MaterializedValidation:
    """Result returned by :func:`validate_materialized`."""

    path: str
    valid: bool
    errors: Tuple[str, ...]
    metadata: Dict[str, object]

    def __bool__(self) -> bool:
        return self.valid


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sampler_description(dataset: ZarrPatchDataset) -> Dict[str, object]:
    sampler = dataset.sampler
    return {
        "type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "parameters": _jsonable(vars(sampler)),
    }


def _configuration(
    dataset: ZarrPatchDataset,
    n: int,
    transform_id: Optional[str],
) -> Dict[str, object]:
    source_path, source_mtime = _source_fingerprint(dataset)
    return {
        "patch_size": dataset.patch_size,
        "variables": list(dataset.variables),
        "dim_slices": {d: list(v) for d, v in dataset._dim_slices.items()},
        "seed": int(dataset.seed),
        "epoch": int(dataset._epoch.value),
        "length": int(n),
        "sampler": _sampler_description(dataset),
        "source": {"path": source_path, "metadata_mtime": source_mtime},
        "transform_id": transform_id,
    }


def _fingerprint(configuration: Dict[str, object]) -> Tuple[str, str]:
    serialized = json.dumps(
        _jsonable(configuration), sort_keys=True, separators=(",", ":")
    )
    return serialized, hashlib.sha256(serialized.encode()).hexdigest()


def validate_materialized(
    path: str,
    *,
    raise_on_error: bool = False,
) -> MaterializedValidation:
    """Validate snapshot format, completion, schema, counts, and fingerprint."""
    h5py = _import_h5py()
    path = os.fspath(path)
    errors: List[str] = []
    metadata: Dict[str, object] = {}
    try:
        with h5py.File(path, "r") as f:
            attrs = f.attrs
            version = int(attrs.get("zarrpatch_format_version", -1))
            if version != MATERIALIZED_FORMAT_VERSION:
                errors.append(
                    f"unsupported zarrpatch materialized format {version}; "
                    f"expected {MATERIALIZED_FORMAT_VERSION} (rebuild snapshot)"
                )
            if not bool(attrs.get("complete", False)):
                errors.append("snapshot is incomplete")
            expected = int(attrs.get("length", -1))
            if "corner" not in f:
                errors.append("missing required 'corner' dataset")
            elif f["corner"].shape[0] != expected:
                errors.append(
                    f"corner count {f['corner'].shape[0]} != expected {expected}"
                )
            try:
                schema = json.loads(attrs["schema"])
            except (KeyError, TypeError, ValueError):
                schema = {}
                errors.append("missing or invalid schema metadata")
            for name, spec in schema.items():
                if name not in f:
                    errors.append(f"missing dataset {name!r}")
                    continue
                actual_shape = list(f[name].shape)
                if actual_shape != spec.get("shape"):
                    errors.append(
                        f"dataset {name!r} shape {actual_shape} != "
                        f"{spec.get('shape')}"
                    )
                if str(f[name].dtype) != spec.get("dtype"):
                    errors.append(
                        f"dataset {name!r} dtype {f[name].dtype} != "
                        f"{spec.get('dtype')}"
                    )
            try:
                config_json = str(attrs["configuration"])
                stored_fingerprint = str(attrs["configuration_sha256"])
                computed = hashlib.sha256(config_json.encode()).hexdigest()
                if computed != stored_fingerprint:
                    errors.append("configuration fingerprint mismatch")
                metadata = json.loads(config_json)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                errors.append("missing or invalid configuration fingerprint")
    except (OSError, ValueError) as exc:
        errors.append(f"cannot open HDF5 snapshot: {exc}")
    result = MaterializedValidation(path, not errors, tuple(errors), metadata)
    if errors and raise_on_error:
        raise ValueError(
            f"invalid zarrpatch materialized snapshot {path!r}: "
            + "; ".join(errors)
        )
    return result


def _human_bytes(n: float) -> str:
    """Format a byte count with a binary unit (``B``/``KiB``/.../``TiB``)."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _source_region_bytes(dataset: ZarrPatchDataset) -> int:
    """Bytes of the source arrays over the (dim-sliced) read region.

    This is roughly what freezing the *whole* covered cube once would cost:
    for every selected variable, the product of its per-dim extents (a
    ``dim_slices`` extent where restricted, else the full dim size) times the
    element size. :func:`materialize` compares the snapshot size against it —
    a snapshot larger than the region it was drawn from means patches overlap
    (or the sampler drew with replacement), so pixels are stored several times.
    Computed from dim sizes only (no disk stat), and independent of any
    ``transform``, since the underlying read is still over ``dataset.variables``.
    """
    reader = dataset.reader
    total = 0
    for v in dataset.variables:
        sizes = reader.dim_sizes(v)
        vol = 1
        for d in reader.dims(v):
            if d in dataset._dim_slices:
                start, stop = dataset._dim_slices[d]
                vol *= stop - start
            else:
                vol *= sizes[d]
        total += vol * np.dtype(reader.dtype(v)).itemsize
    return total


def _import_h5py():
    """Import ``h5py`` lazily with a helpful message when it is missing."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ImportError(
            "materialize/HDF5PatchDataset require the optional 'h5py' "
            "dependency; install it with `pip install zarrpatch[hdf5]` "
            "(or `conda install h5py`)."
        ) from exc
    return h5py


def _sample_spec(
    dataset: ZarrPatchDataset,
) -> Tuple[Dict[str, Tuple[Tuple[int, ...], np.dtype]], int]:
    """Per-variable ``(patch_shape, dtype)`` and the corner column count.

    Derived from the reader so an empty dataset (``N == 0``) can still be
    written. Patched dims take their ``patch_size`` length; sliced non-patched
    dims their restricted extent; all other dims their full extent — matching
    what :meth:`ZarrPatchDataset._slice_group` yields.
    """
    reader = dataset.reader
    specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    for v in dataset.variables:
        sizes = reader.dim_sizes(v)
        shape = []
        for d in reader.dims(v):
            if d in dataset.patch_size:
                shape.append(dataset.patch_size[d])
            elif d in dataset._dim_slices:
                start, stop = dataset._dim_slices[d]
                shape.append(stop - start)
            else:
                shape.append(sizes[d])
        specs[v] = (tuple(shape), np.dtype(reader.dtype(v)))
    return specs, len(dataset.patch_size)


def _source_fingerprint(dataset: ZarrPatchDataset) -> Tuple[str, float]:
    """Best-effort ``(source_path, root_metadata_mtime)`` of the reader.

    Returns ``("", -1.0)`` for readers with no single path (e.g.
    :class:`~zarrpatch.readers.MultiCubeReader`) or when the path cannot be
    stat-ed. The mtime tracks a store's root metadata so a rewrite of the cube
    is visible in the snapshot fingerprint.
    """
    path = getattr(dataset.reader, "path", None)
    if not path:
        return "", -1.0
    mtime = -1.0
    try:
        if os.path.isdir(path):
            for meta in ("zarr.json", ".zgroup", ".zattrs"):
                candidate = os.path.join(path, meta)
                if os.path.exists(candidate):
                    mtime = os.path.getmtime(candidate)
                    break
            else:
                mtime = os.path.getmtime(path)
        elif os.path.exists(path):
            mtime = os.path.getmtime(path)
    except OSError:
        mtime = -1.0
    return str(path), float(mtime)


def materialize(
    dataset: ZarrPatchDataset,
    path: str,
    *,
    compression: Optional[str] = None,
    max_bytes: Optional[int] = None,
    overwrite: bool = False,
    transform_id: Optional[str] = None,
) -> str:
    """Write one sequential pass of ``dataset`` to an HDF5 snapshot file.

    Iterates ``dataset`` once in a single process and writes, per sample key, an
    HDF5 dataset of shape ``(N, *sample_shape)`` chunked one sample per chunk,
    plus an int64 ``corner`` dataset of shape ``(N, len(patch_size))`` whose
    columns are ordered like ``sorted(patch_size)``. ``N`` is the dataset's
    whole epoch (:meth:`~zarrpatch.dataset.ZarrPatchDataset.epoch_size`) at
    its current seed/epoch — also inside an initialized ``torch.distributed``
    job, where the live dataset would otherwise yield only this rank's share.
    Under DDP, call it on one rank, barrier, then read the snapshot on every
    rank. Corners are always recorded, whether or
    not the dataset was built with ``return_corners=True``.

    Without a ``transform`` the layout is derived from the reader (so an empty
    dataset still writes correctly). With a ``transform`` — which may re-key or
    reshape samples arbitrarily, e.g. assemble many raw variables into one
    input tensor — the layout (keys, shapes, dtypes) and the ``max_bytes``
    estimate are taken from the first yielded sample instead; the transform
    must preserve the ``"corner"`` key, and an *empty* transformed dataset
    writes only the (empty) ``corner`` dataset since sample shapes are unknown.

    Parameters
    ----------
    dataset:
        A configured :class:`~zarrpatch.dataset.ZarrPatchDataset`. It is
        iterated as-is (shuffling is harmless — order does not matter for a
        map-style reader), and whatever it is configured to produce (raw or
        transformed patches) is what gets frozen.
    path:
        Output ``.h5`` path.
    compression:
        Passed through to h5py: ``None`` (default, fastest), ``"lzf"`` (a fast
        middle option) or ``"gzip"``.
    max_bytes:
        Pre-flight budget. The estimated total size (per-sample bytes times
        ``N``) is computed before any file is created; if it exceeds
        ``max_bytes`` a :class:`ValueError` is raised and no file is written.
    overwrite:
        When ``False`` (default), refuse to clobber an existing ``path`` with a
        :class:`FileExistsError`.
    transform_id:
        Stable caller-supplied identity for an attached transform. Python
        callables cannot be fingerprinted reliably; when a transform is
        configured this argument is required.

    Returns
    -------
    str
        The ``path`` written.
    """
    h5py = _import_h5py()
    from . import __version__

    path = os.fspath(path)
    if not overwrite and os.path.exists(path):
        raise FileExistsError(
            f"{path!r} exists; pass overwrite=True to replace it"
        )
    if dataset.transform is not None and not transform_id:
        raise ValueError(
            "materializing a transformed dataset requires transform_id=... "
            "because Python callables cannot be fingerprinted reliably"
        )

    # the whole epoch, whatever the DDP topology: a snapshot is a dataset
    # artifact, not one rank's view of it
    prev_single_rank = dataset._single_rank
    dataset._single_rank = True
    try:
        return _materialize(dataset, path, h5py, __version__, compression,
                            max_bytes, overwrite, transform_id)
    finally:
        dataset._single_rank = prev_single_rank


def _materialize(dataset, path, h5py, __version__, compression, max_bytes,
                 overwrite, transform_id) -> str:
    n = len(dataset)
    configuration = _configuration(dataset, n, transform_id)
    configuration_json, configuration_sha256 = _fingerprint(configuration)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    prefix = f".{os.path.basename(path)}."
    fd, temporary_path = tempfile.mkstemp(
        dir=parent, prefix=prefix, suffix=".tmp"
    )
    os.close(fd)

    # force corner output for the build, restoring the flag afterwards
    prev_return_corners = dataset.return_corners
    dataset.return_corners = True
    try:
        if dataset.transform is None:
            specs, n_corner_cols = _sample_spec(dataset)
            samples = iter(dataset)
        else:
            # the transform may re-key/reshape samples, so the layout must
            # come from a real sample: peek the first, then put it back
            samples = iter(dataset)
            first = next(samples, None)
            if first is None:
                specs, n_corner_cols = {}, len(dataset.patch_size)
            else:
                if "corner" not in first:
                    raise ValueError(
                        "the dataset transform dropped the 'corner' key; "
                        "materialize needs it to record patch corners"
                    )
                specs = {
                    k: (np.asarray(a).shape, np.asarray(a).dtype)
                    for k, a in first.items() if k != "corner"
                }
                n_corner_cols = np.asarray(first["corner"]).shape[0]
                samples = chain([first], samples)

        per_sample = sum(prod(shape) * dtype.itemsize
                         for shape, dtype in specs.values())
        per_sample += n_corner_cols * np.dtype(np.int64).itemsize
        estimated = per_sample * n
        if max_bytes is not None and estimated > max_bytes:
            raise ValueError(
                f"materializing {n} samples would need ~{estimated} bytes "
                f"(~{estimated / 2**20:.1f} MiB), over max_bytes={max_bytes} "
                f"(~{max_bytes / 2**20:.1f} MiB); no file written"
            )

        # storage-impact log + duplication warning: `estimated` is the snapshot
        # size; `region` is what a whole-cube freeze of the read region would
        # cost. A ratio > 1 means the snapshot stores more bytes than the source
        # region holds — patches overlap or the sampler drew with replacement —
        # so pixels are written several times.
        region = _source_region_bytes(dataset)
        ratio = estimated / region if region else float("nan")
        _LOG.info(
            "materialize: %d samples -> ~%s (~%s/sample, compression=%s); "
            "source read region ~%s (~%.2gx)",
            n, _human_bytes(estimated), _human_bytes(per_sample),
            compression or "none", _human_bytes(region), ratio,
        )
        if region and estimated > region:
            warnings.warn(
                f"materialize: snapshot ~{_human_bytes(estimated)} exceeds the "
                f"source read region ~{_human_bytes(region)} (~{ratio:.2g}x) — "
                "patches overlap or the sampler draws with replacement, so "
                "pixels are stored multiple times. To avoid the duplication, "
                "freeze a coarser, non-overlapping granularity (drop the "
                "overlapping dim from patch_size and slice it at read time) or "
                "read live.",
                stacklevel=2,
            )

        f = h5py.File(temporary_path, "w")
        with f:
            f.attrs["zarrpatch_format_version"] = MATERIALIZED_FORMAT_VERSION
            f.attrs["complete"] = False
            f.attrs["configuration"] = configuration_json
            f.attrs["configuration_sha256"] = configuration_sha256
            columns = {
                v: f.create_dataset(
                    v,
                    shape=(n, *shape),
                    dtype=dtype,
                    chunks=(1, *shape) if n and shape else None,
                    compression=compression,
                )
                for v, (shape, dtype) in specs.items()
            }
            corner_ds = f.create_dataset(
                "corner",
                shape=(n, n_corner_cols),
                dtype=np.int64,
                chunks=(1, n_corner_cols) if n and n_corner_cols else None,
                compression=compression,
            )

            written = 0
            for i, patch in enumerate(samples):
                if i >= n:
                    raise ValueError(
                        f"dataset yielded more than the expected {n} samples; "
                        "did its seed/epoch change during materialize?"
                    )
                for v in specs:
                    columns[v][i] = np.asarray(patch[v])
                corner_ds[i] = np.asarray(patch["corner"])
                written = i + 1
            if written != n:
                raise ValueError(
                    f"dataset yielded {written} samples but len() reported {n}"
                )

            source_path, source_mtime = _source_fingerprint(dataset)
            attrs = f.attrs
            attrs["patch_size"] = json.dumps(dataset.patch_size)
            attrs["variables"] = json.dumps(list(dataset.variables))
            attrs["dim_slices"] = json.dumps(
                {d: list(bounds) for d, bounds in dataset._dim_slices.items()}
            )
            attrs["corner_columns"] = json.dumps(sorted(dataset.patch_size))
            attrs["seed"] = int(dataset.seed)
            attrs["epoch"] = int(dataset._epoch.value)
            attrs["length"] = int(n)
            attrs["source_path"] = source_path
            attrs["source_mtime"] = source_mtime
            attrs["zarrpatch_version"] = __version__
            attrs["schema"] = json.dumps({
                name: {"shape": list(ds.shape), "dtype": str(ds.dtype)}
                for name, ds in f.items()
            }, sort_keys=True)
            f.flush()
            attrs["complete"] = True
            f.flush()
        validate_materialized(temporary_path, raise_on_error=True)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # Hard-link publication is atomic and cannot clobber a destination
            # created by a competing writer between the preflight and publish.
            os.link(temporary_path, path)
            os.unlink(temporary_path)
    except BaseException:
        dataset.return_corners = prev_return_corners
        try:
            os.remove(temporary_path)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise
    dataset.return_corners = prev_return_corners
    return path


class HDF5PatchDataset(Dataset):
    """Map-style dataset reading a snapshot written by :func:`materialize`.

    ``__getitem__(i)`` returns the same ``Patch`` dict as the live dataset —
    ``{var: np.ndarray, "corner": np.ndarray}`` with the corner columns ordered
    like ``sorted(patch_size)`` — optionally passed through ``transform``.

    The h5py file is opened **lazily per worker**: h5py handles are not
    fork-safe, so the handle is (re)opened inside :meth:`__getitem__` whenever
    the current process id differs from the one it was opened under, and dropped
    when the dataset is pickled into a worker.

    Parameters
    ----------
    path:
        Path to a snapshot ``.h5`` file.
    transform:
        Optional callable applied to each patch dict.
    """

    def __init__(
        self,
        path: str,
        transform: Optional[Callable[[Patch], Patch]] = None,
    ):
        h5py = _import_h5py()
        self.path = os.fspath(path)
        self.transform = transform
        self._file = None
        self._pid: Optional[int] = None
        # read length + fingerprint eagerly, without holding a handle open
        validate_materialized(self.path, raise_on_error=True)
        with h5py.File(self.path, "r") as f:
            self._variables: List[str] = [k for k in f.keys() if k != "corner"]
            self._len = int(f["corner"].shape[0])
            self._fingerprint = {k: f.attrs[k] for k in f.attrs}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    @property
    def _handle(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            h5py = _import_h5py()
            self._file = h5py.File(self.path, "r")
            self._pid = pid
        return self._file

    @property
    def corner_columns(self) -> List[str]:
        """Dim name of each ``corner`` column, in column order.

        E.g. ``reader.corner_columns.index("time")`` gives the column holding
        the time corner — the order is ``sorted(patch_size)``, which is easy
        to get wrong by hand.
        """
        if "corner_columns" in self._fingerprint:
            return list(json.loads(self._fingerprint["corner_columns"]))
        # snapshots written before the attr existed: derive from patch_size
        return sorted(json.loads(self._fingerprint["patch_size"]))

    @property
    def fingerprint(self) -> Dict[str, object]:
        """The root-attribute fingerprint written by :func:`materialize`."""
        return dict(self._fingerprint)

    # alias, mirroring the h5py `.attrs` name
    attrs = fingerprint

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Patch:
        f = self._handle
        patch: Patch = {v: f[v][index] for v in self._variables}
        patch["corner"] = f["corner"][index]
        return self.transform(patch) if self.transform else patch
