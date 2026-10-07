"""Patch dataset with block-grouped reads and a shuffle buffer.

The core idea: reading random patches one by one forces the storage layer to
fetch and decompress a whole chunk (or shard) per patch, so most decompressed
bytes are thrown away. Instead we

1. draw all patch corners for the epoch up front (via a ``Sampler``),
2. group corners by the storage block (shard if present, else chunk) that
   contains them,
3. visit blocks in random order, reading the bounding box of each group's
   patches once per variable and slicing all its patches out of memory,
4. pipe the resulting patch stream through a shuffle buffer to decorrelate
   samples that came from the same block.

``ZarrPatchDataset`` is a ``torch.utils.data.IterableDataset``. The epoch's
shuffled block groups are partitioned across ``torch.distributed`` ranks and
dataloader workers (all ``world_size × num_workers`` shards draw the identical
corner list, then take disjoint, exact-size contiguous pieces of it), so every
patch is yielded exactly once per epoch across the whole job — except that,
under DDP, the ``N mod world_size`` remainder is dropped (or padded, see
``ddp_remainder``) so that every rank yields the same number of batches.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from math import prod
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .readers import CubeReader, open_cube
from .profiling import JsonlProfiler, emit
from .samplers import Corner, GridSampler, Sampler
from .shuffle import shuffled

Patch = Dict[str, np.ndarray]


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader ``worker_init_fn`` that pins each worker to single-threaded
    native pools.

    With ``num_workers=N`` fork workers each importing numpy/numcodecs, every
    worker's BLAS/OpenMP/blosc pool defaults to the machine's core count, so the
    box is oversubscribed ``N x`` — usually a net slowdown, sometimes thrashing.
    Passing ``worker_init_fn=worker_init_fn`` pins torch and blosc to one thread
    per worker, matching what block-grouped, prefetch-pipelined reads want.

    NOTE — this does **not** cure the libomp fork-deadlock (a worker forked
    while a native thread-pool lock is held in the parent). This hook
    runs *after* the fork and after torch's ``_worker_loop`` has already called
    ``set_num_threads(1)`` (the call that can hang). Preventing the deadlock is
    strictly a pre-fork action: set ``OMP_NUM_THREADS=1`` before launch and/or
    build the loader with ``multiprocessing_context="forkserver"`` (or
    ``"spawn"``). Use this hook for oversubscription; use those for the hang.
    """
    import torch

    torch.set_num_threads(1)
    try:  # blosc keeps its own pool, independent of torch's
        import numcodecs

        numcodecs.blosc.set_nthreads(1)
    except Exception:  # numcodecs may be absent or blosc-less  # pragma: no cover
        pass


def group_corners_by_block(corners: Sequence[Corner],
                           block_size: Dict[str, int],
                           offsets: Optional[Dict[str, int]] = None,
                           ) -> List[List[Corner]]:
    """Group patch corners by the storage block containing them.

    ``offsets`` shifts corners into global index space before computing the
    block, so grouping stays aligned to storage blocks when the corners live
    in a sliced index space (see ``ZarrPatchDataset(dim_slices=...)``).
    """
    groups: Dict[Tuple[int, ...], List[Corner]] = {}
    dims = list(block_size)
    offsets = offsets or {}
    for corner in corners:
        key = tuple((corner.get(d, 0) + offsets.get(d, 0)) // block_size[d]
                    for d in dims)
        groups.setdefault(key, []).append(corner)
    return [groups[k] for k in sorted(groups)]


def _bounding_region(group: Sequence[Corner],
                     patch_size: Dict[str, int]) -> Dict[str, slice]:
    region = {}
    for d, p in patch_size.items():
        starts = [c[d] for c in group]
        region[d] = slice(min(starts), max(starts) + p)
    return region


def _dist_info() -> Tuple[int, int]:
    """(rank, world_size) under an initialized torch.distributed, else (0, 1)."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return 0, 1
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


DEFAULT_SHUFFLE_BUFFER = 2048
BUFFER_MODES = ("auto", "io")


def resolve_shuffle_buffer(spec: Union[int, str],
                           groups: Sequence[Sequence[Corner]]) -> int:
    """Resolve a ``shuffle_buffer`` spec against one shard's block groups."""
    if spec in BUFFER_MODES:
        io = 4 * max((len(g) for g in groups), default=0)
        return io if spec == "io" else max(DEFAULT_SHUFFLE_BUFFER, io)
    return int(spec)


def _even_sizes(total: int, n: int) -> List[int]:
    """``n`` sizes summing to ``total`` that differ by at most one."""
    q, r = divmod(total, n)
    return [q + 1 if i < r else q for i in range(n)]


def _split_contiguous(groups: Sequence[Sequence[Corner]],
                      sizes: Sequence[int]) -> List[List[List[Corner]]]:
    """Cut the patch sequence of ``groups`` into consecutive exact-size pieces.

    Groups are kept whole where possible; a group straddling a cut is split
    between two pieces (both then read that block — at most ``len(sizes) - 1``
    extra group reads). Patches beyond ``sum(sizes)`` are dropped.
    """
    pieces: List[List[List[Corner]]] = []
    it = iter(groups)
    rest: List[Corner] = []
    for size in sizes:
        piece: List[List[Corner]] = []
        need = size
        while need > 0:
            if not rest:
                rest = list(next(it))
            take, rest = rest[:need], rest[need:]
            piece.append(take)
            need -= len(take)
        pieces.append(piece)
    return pieces


def _rank_groups(groups: List[List[Corner]], world: int,
                 remainder: str) -> List[List[List[Corner]]]:
    """Split an epoch's groups into ``world`` pieces of *identical* size.

    ``remainder="drop"`` drops the ``N mod world`` last patches;
    ``"pad"`` re-yields the first ``(-N) mod world`` patches at the end.
    """
    total = sum(map(len, groups))
    if world == 1:
        return [groups]
    if remainder == "pad":
        extra = -total % world
        if extra:
            groups = list(groups) + _split_contiguous(groups, [extra])[0]
        per = (total + extra) // world
    else:
        per = total // world
    return _split_contiguous(groups, [per] * world)


def _shard_groups(groups: List[List[Corner]], rank: int, world: int,
                  wid: int, nw: int, remainder: str) -> List[List[Corner]]:
    """Groups for one (rank, worker) shard: rank split, then worker split.

    Every rank gets exactly the same patch count, and — for the same
    ``num_workers`` — the same per-worker counts, so every rank yields the
    same number of batches (a DDP requirement: a rank that finishes early
    deadlocks the others' collective ops).
    """
    mine = _rank_groups(groups, world, remainder)[rank]
    sizes = _even_sizes(sum(map(len, mine)), nw)
    return _split_contiguous(mine, sizes)[wid]


class ZarrPatchDataset(IterableDataset):
    """Iterable dataset of patches from a zarr or netCDF datacube.

    Parameters
    ----------
    path:
        Path to the zarr store or netCDF file, or an already-constructed
        :class:`~zarrpatch.readers.CubeReader` (e.g. a
        :class:`~zarrpatch.readers.MultiCubeReader`), used directly.
    patch_size:
        Dim name -> patch length. Dims of a variable not listed here are read
        in full extent.
    variables:
        Variables to load; defaults to every variable containing all patch dims.
    sampler:
        A :class:`~zarrpatch.samplers.Sampler`; defaults to a non-overlapping
        :class:`~zarrpatch.samplers.GridSampler` covering the whole cube.
    backend:
        ``'zarr'``, ``'tensorstore'``, ``'netcdf'``, ``'kerchunk'`` or
        ``None`` (guess from extension; zarr stores default to ``'zarr'``).
    group_by_block:
        Group patch reads by storage block (the main optimization). Disable
        only for benchmarking the naive per-patch baseline.
    shuffle:
        Randomize block visit order each epoch.
    shuffle_buffer:
        Size of the streaming shuffle buffer in patches (0/1 disables it). The
        buffer holds that many decoded patches in memory per worker, and it
        bounds batch-level mixing: with an ``IterableDataset`` each batch is
        collated from *one* worker's stream, so a batch only mixes patches
        that were in that worker's buffer. Symbolic sizes:

        - ``"auto"``: ``max(2048, 4 × largest patches-per-block)`` — never
          smaller than the default, grown when blocks are dense.
        - ``"io"``: ``4 × largest patches-per-block`` only — the minimum that
          stops one decoded block landing whole in one batch. It is an I/O
          de-correlation rule, **not** a mixing guarantee: with few patches
          per block it can be tiny (e.g. 48). Use ``plan(batch_size=...)`` to
          see the mixing you actually get.
    seed:
        Base RNG seed; combined with the epoch (see :meth:`set_epoch`) and, for
        block order and the buffer, the shard id (rank x worker).
    transform:
        Optional callable applied to each patch dict.
    return_corners:
        Also yield the patch corner under the ``"corner"`` key, as an int64
        array ordered like ``sorted(patch_size)``.
    prefetch:
        Number of block groups to read ahead in a background thread while the
        current group's patches are being sliced and consumed (0 disables).
    max_block_bytes:
        Cap on the estimated bytes read per block group (all variables,
        bounding region). Storage blocks that would exceed it — e.g. a
        contiguous netCDF variable, whose "block" is the whole array — are
        split into smaller virtual blocks, with a warning. ``None`` disables.
    ddp_remainder:
        Under ``torch.distributed`` (``world_size > 1``) every rank must yield
        the same number of batches, or ranks that finish early deadlock the
        others' collective ops at epoch end. Ranks therefore get exactly equal
        patch counts: ``"drop"`` (default) skips the last ``N mod world_size``
        patches of the epoch, ``"pad"`` re-yields the first
        ``(-N) mod world_size`` ones. Worker counts within a rank are then
        identical across ranks too. No effect without DDP.
    group_block:
        Dim name -> edge of a *virtual* grouping block along patched dims,
        overriding the storage block (shard, else chunk). Each edge must be a
        multiple of the chunk edge. A group's read region extends
        ``patch - 1`` past its block, so grouping by a single chunk whose edge
        is close to the patch edge decodes most chunks up to ``2**d`` times;
        grouping by several chunks (what a zarr v3 shard does) amortizes that
        straddle. A warning suggests a value when the default grouping is
        likely to straddle heavily.
    dim_slices:
        Dim name -> ``slice(start, stop)`` or ``(start, stop)`` restricting
        the dataset to that index range — e.g. a train/val/test split of a
        single store along time (``{"time": (0, 800)}``), no per-split copies
        needed. The dataset then behaves as if the cube were sliced: sampler
        corners (including ``ListSampler`` input and the ``"corner"`` output)
        are in the *sliced* index space. Sliced dims not in ``patch_size``
        are read over the restricted range instead of full extent.
    """

    def __init__(
        self,
        path: Union[str, CubeReader],
        patch_size: Dict[str, int],
        variables: Optional[Sequence[str]] = None,
        sampler: Optional[Sampler] = None,
        backend: Optional[str] = None,
        group_by_block: bool = True,
        shuffle: bool = True,
        shuffle_buffer: Union[int, str] = DEFAULT_SHUFFLE_BUFFER,
        seed: int = 0,
        transform: Optional[Callable[[Patch], Patch]] = None,
        return_corners: bool = False,
        prefetch: int = 1,
        max_block_bytes: Optional[int] = 1024 * 1024 * 1024,
        dim_slices: Optional[Dict[str, Union[slice, Tuple[int, int]]]] = None,
        profiler: Optional[JsonlProfiler] = None,
        ddp_remainder: str = "drop",
        group_block: Optional[Dict[str, int]] = None,
    ):
        self.reader: CubeReader = (
            path if isinstance(path, CubeReader)
            else open_cube(path, backend, profiler=profiler)
        )
        self.profiler = profiler
        self.reader.set_profiler(profiler)
        self.patch_size = dict(patch_size)
        check = getattr(self.reader, "check_patch_size", None)
        if check is not None:
            check(self.patch_size)
        self.sampler = sampler if sampler is not None else GridSampler()
        self.group_by_block = group_by_block
        self.shuffle = shuffle
        if (isinstance(shuffle_buffer, str)
                and shuffle_buffer not in BUFFER_MODES):
            raise ValueError(f"shuffle_buffer must be an int, 'auto' or 'io', "
                             f"got {shuffle_buffer!r}")
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.transform = transform
        self.return_corners = return_corners
        self.prefetch = prefetch
        self.max_block_bytes = max_block_bytes
        if ddp_remainder not in ("drop", "pad"):
            raise ValueError(f"ddp_remainder must be 'drop' or 'pad', "
                             f"got {ddp_remainder!r}")
        self.ddp_remainder = ddp_remainder
        self._len_cache: Dict[int, int] = {}
        # set by materialize(): treat this process as the only rank, so a
        # snapshot built inside an initialized DDP job holds the whole epoch
        self._single_rank = False
        # shared across processes so set_epoch() in the main process reaches
        # dataloader workers even with persistent_workers=True; the spawn
        # context makes it picklable into workers of either start method
        self._epoch = mp.get_context("spawn").Value("l", 0)

        if variables is None:
            variables = [
                v for v in self.reader.variables
                if set(self.patch_size) <= set(self.reader.dims(v))
            ]
            if not variables:
                raise ValueError(
                    f"no variable has all patch dims {sorted(self.patch_size)}"
                )
        self.variables = list(variables)

        # dim -> (start, stop) restriction in global index space
        self._dim_slices: Dict[str, Tuple[int, int]] = {}
        if dim_slices:
            all_sizes: Dict[str, int] = {}
            for v in self.variables:
                all_sizes.update(self.reader.dim_sizes(v))
            for d, s in dim_slices.items():
                if d not in all_sizes:
                    raise ValueError(
                        f"dim_slices dim {d!r} not in the dims of the "
                        f"selected variables {sorted(all_sizes)}"
                    )
                if isinstance(s, slice):
                    if s.step not in (None, 1):
                        raise ValueError(f"dim_slices[{d!r}]: step not supported")
                    start = 0 if s.start is None else s.start
                    stop = all_sizes[d] if s.stop is None else s.stop
                else:
                    start, stop = s
                if not 0 <= start < stop <= all_sizes[d]:
                    raise ValueError(
                        f"dim_slices[{d!r}] = ({start}, {stop}) out of bounds "
                        f"for dim of size {all_sizes[d]}"
                    )
                self._dim_slices[d] = (int(start), int(stop))

        # dim sizes must agree across variables for the patched dims
        self._dim_sizes: Dict[str, int] = {}
        for v in self.variables:
            for d, n in self.reader.dim_sizes(v).items():
                if d in self.patch_size:
                    if self._dim_sizes.setdefault(d, n) != n:
                        raise ValueError(f"inconsistent size for dim {d!r}")
        for d, (start, stop) in self._dim_slices.items():
            if d in self._dim_sizes:
                self._dim_sizes[d] = stop - start

        # block size along patched dims, from the first variable's layout
        ref = self.variables[0]
        blocks = dict(zip(self.reader.dims(ref), self.reader.block_shape(ref)))
        self._block_size = {d: blocks[d] for d in self.patch_size}
        chunks = dict(zip(self.reader.dims(ref), self.reader.chunk_shape(ref)))
        self._chunk_size = {d: chunks[d] for d in self.patch_size}
        if group_block:
            for d, edge in group_block.items():
                if d not in self.patch_size:
                    raise ValueError(f"group_block dim {d!r} is not a patch dim")
                c = self._chunk_size[d]
                if edge < 1 or edge % c:
                    raise ValueError(
                        f"group_block[{d!r}]={edge} must be a positive multiple "
                        f"of the chunk edge {c}"
                    )
                self._block_size[d] = int(edge)
        for v in self.variables[1:]:
            other = dict(zip(self.reader.dims(v), self.reader.block_shape(v)))
            mismatched = {d: other[d] for d in self.patch_size
                          if d in other and other[d] != self._block_size[d]}
            if mismatched:
                warnings.warn(
                    f"variable {v!r} has block sizes {mismatched} but reads "
                    f"are grouped by {ref!r}'s layout "
                    f"({dict(self._block_size)}); {v!r} may be decoded "
                    "multiple times per epoch"
                )
        self._cap_block_size()
        if not group_block:
            self._warn_chunk_straddle()

    def _dense_decode_factor(self, block_size: Dict[str, int]) -> float:
        """Upper bound on decoded / unique bytes for densely sampled groups.

        A group's read region starts in its block and extends ``patch - 1``
        past it, so it touches ``ceil((block + patch - 1) / chunk)`` chunks
        per dim; every chunk is also decoded by the neighbouring group.
        """
        factor = 1.0
        for d, b in block_size.items():
            c, p, n = self._chunk_size[d], self.patch_size[d], self._dim_sizes[d]
            b = min(b, n)
            span = min(b + p - 1, n)
            factor *= -(-span // c) * c / b
        return factor

    def _warn_chunk_straddle(self) -> None:
        factor = self._dense_decode_factor(self._block_size)
        if factor < 4:
            return
        suggest = {}
        for d, c in self._chunk_size.items():
            p = self.patch_size[d]
            # smallest chunk multiple whose straddle overhead is <= 25%
            suggest[d] = max(self._block_size[d], c * -(-4 * (p - 1) // c))
        warnings.warn(
            f"reads are grouped by blocks {dict(self._block_size)} with patch "
            f"{self.patch_size}: a densely sampled group's region straddles "
            f"the next chunk on every patched dim, so chunks may be decoded up "
            f"to ~{factor:.1f}x per epoch. Group by several chunks instead — "
            f"zarr v3 shards, or group_block={suggest} (costs more memory per "
            "group read). ds.plan() reports the actual decoded bytes."
        )

    def _rank_world(self) -> Tuple[int, int]:
        """(rank, world_size) used to split the epoch; see ``_single_rank``."""
        return (0, 1) if self._single_rank else _dist_info()

    def _offset(self, d: str) -> int:
        """Global index of local index 0 along a (possibly sliced) dim."""
        return self._dim_slices.get(d, (0, 0))[0]

    def _group_read_bytes(self, block_size: Dict[str, int]) -> int:
        """Estimated bytes read for one block group's bounding region."""
        total = 0
        for v in self.variables:
            sizes = self.reader.dim_sizes(v)
            for d, (start, stop) in self._dim_slices.items():
                if d in sizes:
                    sizes[d] = stop - start
            extents = (
                min(block_size[d], sizes[d]) + self.patch_size[d] - 1
                if d in self.patch_size else sizes[d]
                for d in self.reader.dims(v)
            )
            total += prod(extents) * self.reader.dtype(v).itemsize
        return total

    def _cap_block_size(self) -> None:
        if self.max_block_bytes is None:
            return
        capped = dict(self._block_size)
        while self._group_read_bytes(capped) > self.max_block_bytes:
            shrinkable = [d for d in capped if capped[d] > self.patch_size[d]]
            if not shrinkable:
                break
            d = max(shrinkable, key=lambda d: capped[d] / self.patch_size[d])
            p = self.patch_size[d]
            capped[d] = max(p, capped[d] // 2 // p * p)
        if capped != self._block_size:
            warnings.warn(
                f"storage blocks {self._block_size} would read "
                f"~{self._group_read_bytes(self._block_size) // 2**20} MiB per "
                f"group (> max_block_bytes); splitting into virtual blocks "
                f"{capped}. Typical for contiguous netCDF; for compressed "
                "stores this re-decodes chunks and a larger max_block_bytes "
                "may be faster."
            )
            self._block_size = capped

    def set_epoch(self, epoch: int) -> None:
        """Change the sampling/shuffling RNG stream (call once per epoch).

        The epoch lives in shared memory, so this also reaches dataloader
        workers kept alive by ``persistent_workers=True``.
        """
        self._epoch.value = epoch

    def epoch_size(self, epoch: Optional[int] = None) -> int:
        """Total patches drawn for an epoch, across all ranks (cached)."""
        epoch = int(self._epoch.value if epoch is None else epoch)
        if epoch not in self._len_cache:
            rng = np.random.default_rng((self.seed, epoch))
            self._len_cache[epoch] = len(
                self.sampler.sample(self._dim_sizes, self.patch_size, rng))
        return self._len_cache[epoch]

    def __len__(self) -> int:
        """Patches this process's rank yields per epoch.

        Without DDP this is the whole epoch (:meth:`epoch_size`). Under
        ``torch.distributed`` it is the per-rank count — like
        ``DistributedSampler`` — which is what Lightning and LR schedulers
        divide by the batch size: ``N // world_size`` for
        ``ddp_remainder="drop"``, ``ceil(N / world_size)`` for ``"pad"``.
        """
        total = self.epoch_size()
        _, world = self._rank_world()
        if self.ddp_remainder == "pad":
            return -(-total // world)
        return total // world

    def plan(self, *, epoch: Optional[int] = None, world_size: int = 1,
             num_workers: int = 1, batch_size: Optional[int] = None):
        """Inspect this dataset's exact execution plan without reading payloads.

        Pass the DataLoader's ``batch_size`` to also get batch-mixing
        statistics (see :func:`zarrpatch.planning.plan_dataset`).
        """
        from .planning import plan_dataset

        return plan_dataset(
            self, epoch=epoch, world_size=world_size, num_workers=num_workers,
            batch_size=batch_size,
        )

    # -- iteration -----------------------------------------------------------

    def _read_group(self, group: Sequence[Corner]):
        start_ns = time.monotonic_ns()
        emit(self.profiler, "group_read_start", detail=True,
             patches=len(group))
        local = _bounding_region(group, self.patch_size)
        # shift patched dims to global index space; restrict sliced
        # non-patched dims (otherwise read in full extent)
        region = {d: slice(s.start + self._offset(d), s.stop + self._offset(d))
                  for d, s in local.items()}
        for d, (start, stop) in self._dim_slices.items():
            if d not in region:
                region[d] = slice(start, stop)
        buffers = self.reader.read_many(self.variables, region)
        emit(self.profiler, "group_read_end", detail=True, patches=len(group),
             bytes=sum(a.nbytes for a in buffers.values()),
             elapsed_ns=time.monotonic_ns() - start_ns)
        return group, local, buffers

    def _read_ahead(self, groups: Sequence[Sequence[Corner]]) -> Iterator:
        """Yield ``(group, region, buffers)``, reading ahead in a thread.

        A single reader thread keeps ``prefetch`` groups in flight, so
        decompression of the next block overlaps with slicing/consuming the
        current one (and backends need not be thread-safe across threads).
        """
        if self.prefetch <= 0:
            for g in groups:
                yield self._read_group(g)
            return
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = deque()
            it = iter(groups)
            for g in islice(it, self.prefetch):
                pending.append(pool.submit(self._read_group, g))
            for g in it:
                done = pending.popleft()
                pending.append(pool.submit(self._read_group, g))
                yield done.result()
            while pending:
                yield pending.popleft().result()

    def _slice_group(self, group, region, buffers) -> Iterator[Patch]:
        for corner in group:
            patch: Patch = {}
            for v in self.variables:
                local = tuple(
                    slice(corner[d] - region[d].start,
                          corner[d] - region[d].start + self.patch_size[d])
                    if d in self.patch_size else slice(None)
                    for d in self.reader.dims(v)
                )
                patch[v] = np.ascontiguousarray(buffers[v][local])
            if self.return_corners:
                patch["corner"] = np.array(
                    [corner[d] for d in sorted(self.patch_size)], dtype=np.int64
                )
            yield self.transform(patch) if self.transform else patch

    def __iter__(self) -> Iterator[Patch]:
        # same seed in every shard: all shards draw the identical corner list
        # and block grouping, then take disjoint block subsets
        epoch = self._epoch.value
        emit(self.profiler, "dataset_iter_start", epoch=epoch)
        plan_start = time.monotonic_ns()
        rng = np.random.default_rng((self.seed, epoch))
        corners = self.sampler.sample(self._dim_sizes, self.patch_size, rng)

        if self.group_by_block:
            offsets = {d: self._offset(d) for d in self._block_size}
            groups = group_corners_by_block(corners, self._block_size, offsets)
        else:
            groups = [[c] for c in corners]

        if self.shuffle:
            rng.shuffle(groups)

        rank, world = self._rank_world()
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        shard, n_shards = rank * nw + wid, world * nw
        groups = _shard_groups(groups, rank, world, wid, nw,
                               self.ddp_remainder)
        emit(self.profiler, "dataset_plan_end", epoch=epoch, worker_id=wid,
             workers=nw, shard=shard, shards=n_shards, corners=len(corners),
             groups=len(groups),
             elapsed_ns=time.monotonic_ns() - plan_start)

        shard_rng = np.random.default_rng((self.seed, epoch, shard))
        if self.shuffle:
            shard_rng.shuffle(groups)

        buffer = resolve_shuffle_buffer(self.shuffle_buffer, groups)

        stream = (p for grb in self._read_ahead(groups)
                  for p in self._slice_group(*grb))
        if self.shuffle and buffer > 1:
            stream = shuffled(stream, buffer, shard_rng)
        first = True
        yielded = 0
        for patch in stream:
            if first:
                emit(self.profiler, "first_patch", epoch=epoch, worker_id=wid,
                     shard=shard)
                first = False
            yielded += 1
            yield patch
        emit(self.profiler, "dataset_iter_end", epoch=epoch, worker_id=wid,
             shard=shard, patches=yielded)
