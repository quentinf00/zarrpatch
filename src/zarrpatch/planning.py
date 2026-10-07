"""Read-free inspection of a :class:`ZarrPatchDataset` execution plan."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import prod
from os import PathLike
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .dataset import (
    ZarrPatchDataset,
    _bounding_region,
    _shard_groups,
    group_corners_by_block,
    resolve_shuffle_buffer,
)
from .samplers import Corner
from .shuffle import shuffled


@dataclass(frozen=True)
class ExecutionPlan:
    """Machine-readable, exact description of one dataset epoch's layout.

    ``region_bytes`` is the sum of the bounding regions actually read (the
    arrays returned by the reader); ``decoded_bytes`` expands every region to
    whole chunks, i.e. what the codecs decompress, and ``read_amplification``
    is ``decoded_bytes / logical_bytes``. ``batch_mixing`` is only filled
    when a ``batch_size`` is given.
    """

    exact: bool
    epoch: int
    patches: int
    yielded_patches: int
    groups: int
    patches_per_group: Dict[str, float]
    shard_patch_counts: List[int]
    shard_group_counts: List[int]
    logical_patch_bytes: int
    logical_bytes: int
    region_bytes: int
    decoded_bytes: int
    read_amplification: float
    bounding_region_bytes: Dict[str, float]
    shuffle_buffer_patches: int
    estimated_inflight_bytes_per_worker: int
    reference_variable: str
    variable_block_shapes: Dict[str, Dict[str, int]]
    grouping_block: Dict[str, int]
    incompatible_variables: List[str]
    estimated_store_opens: int
    batch_mixing: Optional[Dict[str, float]]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self, path: Optional[Union[str, PathLike]] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        return text

    def summary(self) -> str:
        ppg = self.patches_per_group
        lines = [
            f"{self.patches:,} patches in {self.groups:,} physical groups "
            f"(exact={self.exact}); {self.yielded_patches:,} yielded",
            f"patches/group: min={ppg['min']:.0f}, median={ppg['p50']:.1f}, "
            f"p95={ppg['p95']:.1f}, max={ppg['max']:.0f}",
            f"logical={self.logical_bytes / 2**20:.1f} MiB, "
            f"read={self.region_bytes / 2**20:.1f} MiB, "
            f"decoded={self.decoded_bytes / 2**20:.1f} MiB, "
            f"amplification={self.read_amplification:.2f}x",
            f"per-shard patches={self.shard_patch_counts}",
            f"shuffle buffer={self.shuffle_buffer_patches} patches; "
            f"estimated in-flight memory/worker="
            f"{self.estimated_inflight_bytes_per_worker / 2**20:.1f} MiB",
        ]
        if self.batch_mixing:
            m = self.batch_mixing
            lines.append(
                f"batch mixing (batch {m['batch_size']:.0f}): "
                f"{m['distinct_groups_mean']:.1f} distinct groups/batch "
                f"(min {m['distinct_groups_min']:.0f}), largest single-group "
                f"share {m['largest_group_share_mean']:.0%} mean / "
                f"{m['largest_group_share_max']:.0%} max (ideal global "
                f"shuffle: {m['iid_distinct_groups_mean']:.1f} groups, "
                f"{m['iid_largest_group_share_mean']:.0%})"
            )
        lines.extend(f"WARNING: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _stats(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    a = np.asarray(values)
    return {
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _extent(dataset: ZarrPatchDataset, variable: str,
            region: Optional[Dict[str, slice]], chunk_aligned: bool) -> int:
    """Bytes of ``variable`` over a local-space region (None = one patch)."""
    reader = dataset.reader
    sizes = reader.dim_sizes(variable)
    chunks = dict(zip(reader.dims(variable), reader.chunk_shape(variable)))
    extents = []
    for dim in reader.dims(variable):
        offset = dataset._offset(dim)
        if dim in dataset.patch_size:
            if region is None:
                a, b = 0, dataset.patch_size[dim]
            else:
                a, b = region[dim].start, region[dim].stop
            a, b = a + offset, b + offset
        elif dim in dataset._dim_slices:
            a, b = dataset._dim_slices[dim]
        else:
            a, b = 0, sizes[dim]
        if chunk_aligned:
            c = chunks[dim]
            a, b = a // c * c, min(-(-b // c) * c, sizes[dim])
        extents.append(b - a)
    return prod(extents) * reader.dtype(variable).itemsize


def _batch_mixing(dataset: ZarrPatchDataset, groups: List[List[Corner]],
                  epoch: int, world_size: int, num_workers: int,
                  batch_size: int) -> Dict[str, float]:
    """Replay every shard's exact patch stream on group labels.

    Mirrors ``ZarrPatchDataset.__iter__`` (same split, same RNG draws), so
    the statistics are those of the batches a DataLoader would collate —
    each from a single worker's stream.
    """
    labels = [[gi] * len(g) for gi, g in enumerate(groups)]

    def stats(seq: np.ndarray, distinct: list, largest: list) -> None:
        for i in range(0, len(seq), batch_size):
            batch = seq[i:i + batch_size]
            counts = np.unique(batch, return_counts=True)[1]
            distinct.append(len(counts))
            largest.append(counts.max() / len(batch))

    distinct, largest = [], []
    for rank in range(world_size):
        for wid in range(num_workers):
            shard = rank * num_workers + wid
            mine = _shard_groups(labels, rank, world_size, wid, num_workers,
                                 dataset.ddp_remainder)
            rng = np.random.default_rng((dataset.seed, epoch, shard))
            if dataset.shuffle:
                rng.shuffle(mine)
            buffer = resolve_shuffle_buffer(dataset.shuffle_buffer, mine)
            stream = (label for g in mine for label in g)
            if dataset.shuffle and buffer > 1:
                stream = shuffled(stream, buffer, rng)
            stats(np.fromiter(stream, dtype=np.int64), distinct, largest)
    if not distinct:
        return {}
    # reference: the same patches in a global uniform permutation, i.e. what
    # an ideal map-style shuffle over the whole epoch would give
    iid_distinct, iid_largest = [], []
    everything = np.fromiter((x for g in labels for x in g), dtype=np.int64)
    stats(np.random.default_rng(0).permutation(everything),
          iid_distinct, iid_largest)
    return {
        "batch_size": float(batch_size),
        "batches": float(len(distinct)),
        "distinct_groups_mean": float(np.mean(distinct)),
        "distinct_groups_min": float(np.min(distinct)),
        "largest_group_share_mean": float(np.mean(largest)),
        "largest_group_share_max": float(np.max(largest)),
        "iid_distinct_groups_mean": float(np.mean(iid_distinct)),
        "iid_largest_group_share_mean": float(np.mean(iid_largest)),
    }


def plan_dataset(
    dataset: ZarrPatchDataset,
    *,
    epoch: Optional[int] = None,
    world_size: int = 1,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> ExecutionPlan:
    """Build an exact plan using metadata and sampler corners, never payloads.

    ``batch_size`` additionally replays the shuffle stream of every shard to
    report how well batches mix block groups (cost O(patches)).
    """
    if world_size < 1 or num_workers < 1:
        raise ValueError("world_size and num_workers must be >= 1")
    epoch = int(dataset._epoch.value if epoch is None else epoch)
    rng = np.random.default_rng((dataset.seed, epoch))
    corners = dataset.sampler.sample(dataset._dim_sizes, dataset.patch_size, rng)
    if dataset.group_by_block:
        offsets = {d: dataset._offset(d) for d in dataset._block_size}
        groups = group_corners_by_block(corners, dataset._block_size, offsets)
    else:
        groups = [[corner] for corner in corners]
    if dataset.shuffle:  # same draw as __iter__, so the split matches exactly
        rng.shuffle(groups)

    shards = [
        _shard_groups(groups, rank, world_size, wid, num_workers,
                      dataset.ddp_remainder)
        for rank in range(world_size) for wid in range(num_workers)
    ]
    group_lengths = [len(group) for group in groups]
    logical_patch_bytes = sum(
        _extent(dataset, variable, None, chunk_aligned=False)
        for variable in dataset.variables
    )

    region_bytes, decoded = [], 0
    for shard in shards:
        for group in shard:
            region = _bounding_region(group, dataset.patch_size)
            region_bytes.append(sum(
                _extent(dataset, v, region, chunk_aligned=False)
                for v in dataset.variables))
            decoded += sum(_extent(dataset, v, region, chunk_aligned=True)
                           for v in dataset.variables)
    yielded = sum(len(g) for shard in shards for g in shard)

    block_shapes = {
        variable: dict(zip(dataset.reader.dims(variable),
                           dataset.reader.block_shape(variable)))
        for variable in dataset.variables
    }
    reference = dataset.variables[0]
    incompatible = [
        variable for variable in dataset.variables[1:]
        if any(block_shapes[variable].get(dim) != dataset._block_size[dim]
               for dim in dataset.patch_size
               if dim in block_shapes[variable])
    ]
    warnings = []
    if incompatible:
        warnings.append(
            f"variables {incompatible} do not match reference variable "
            f"{reference!r}'s grouping layout"
        )
    amplification = (
        decoded / (yielded * logical_patch_bytes)
        if yielded and logical_patch_bytes else 0.0
    )
    if amplification >= 4:
        warnings.append(
            f"estimated decode amplification is {amplification:.1f}x; group "
            "by several chunks (zarr v3 shards or group_block=...), align "
            "storage blocks with patch geometry, or draw more patches per block"
        )
    buffers = [resolve_shuffle_buffer(dataset.shuffle_buffer, shard)
               for shard in shards]
    buffer = max(buffers, default=0)
    largest_region = max(region_bytes, default=0)
    inflight = max(buffer, 0) * logical_patch_bytes
    inflight += (1 + max(dataset.prefetch, 0)) * largest_region
    reader = dataset.reader
    sources = getattr(reader, "_sources", None)
    store_opens = len({path for path, _ in sources}) if sources else 1

    mixing = None
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        mixing = _batch_mixing(dataset, groups, epoch, world_size,
                               num_workers, batch_size) or None
        if mixing and (mixing["largest_group_share_mean"]
                       >= 2 * mixing["iid_largest_group_share_mean"] + 0.05):
            warnings.append(
                f"poor batch mixing: on average "
                f"{mixing['largest_group_share_mean']:.0%} of a batch comes "
                "from a single block group, versus "
                f"{mixing['iid_largest_group_share_mean']:.0%} for an ideal "
                "global shuffle of the same epoch (each batch is collated from "
                "one worker's stream); raise shuffle_buffer (resolved "
                f"{buffer}) or draw fewer patches per block"
            )

    return ExecutionPlan(
        exact=True,
        epoch=epoch,
        patches=len(corners),
        yielded_patches=yielded,
        groups=len(groups),
        patches_per_group=_stats(group_lengths),
        shard_patch_counts=[sum(map(len, shard)) for shard in shards],
        shard_group_counts=[len(shard) for shard in shards],
        logical_patch_bytes=logical_patch_bytes,
        logical_bytes=yielded * logical_patch_bytes,
        region_bytes=sum(region_bytes),
        decoded_bytes=decoded,
        read_amplification=amplification,
        bounding_region_bytes=_stats(region_bytes),
        shuffle_buffer_patches=buffer,
        estimated_inflight_bytes_per_worker=inflight,
        reference_variable=reference,
        variable_block_shapes=block_shapes,
        grouping_block=dict(dataset._block_size),
        incompatible_variables=incompatible,
        estimated_store_opens=store_opens,
        batch_mixing=mixing,
        warnings=warnings,
    )


def analyze_layout(
    path,
    patch_size,
    *,
    world_size: int = 1,
    num_workers: int = 1,
    epoch: int = 0,
    batch_size: Optional[int] = None,
    **dataset_kwargs,
) -> ExecutionPlan:
    """Open an existing store and inspect its workload layout without reads."""
    dataset = ZarrPatchDataset(path, patch_size, **dataset_kwargs)
    return plan_dataset(
        dataset, epoch=epoch, world_size=world_size, num_workers=num_workers,
        batch_size=batch_size,
    )
