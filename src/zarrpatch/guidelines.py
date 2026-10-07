"""Chunk/shard layout recommendations for patch-based training.

The rules implemented here (see also the README):

- **Chunk edge ~ patch edge** along patched dims. Much smaller chunks multiply
  request/decompress overhead; much larger chunks waste decompressed bytes when
  patches are read individually and inflate the shuffle-buffer size needed to
  decorrelate block-grouped reads.
- For **grid/stride sampling**, make the chunk edge a multiple of the stride so
  patches align with chunk boundaries and no chunk is read twice.
- **Un-patched dims** (read in full for every patch) should be chunked so the
  total chunk stays in the ~1-16 MiB sweet spot for parallel decompression.
- **Shards (zarr v3)**: keep the chunk small for read granularity, then pack
  many chunks per shard file to keep the *file count* low — on HPC parallel
  filesystems (Lustre/GPFS) millions of tiny chunk files destroy metadata
  performance. Target ~64-512 MiB per shard.
- **Group reads by several chunks.** A block group's read region extends
  ``patch - 1`` past its block, so grouping by *one* patch-sized chunk
  straddles the next chunk on every patched dim and decodes chunks up to
  ``2**d`` times. Group by a block of at least ~``4 × (patch - 1)`` per patched
  dim — the shard when sharded, else ``ZarrPatchDataset(group_block=...)`` —
  to keep that straddle under ~25%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, prod
from typing import Dict, List, Optional

MiB = 1024 * 1024


@dataclass
class LayoutRecommendation:
    chunks: Dict[str, int]
    shards: Optional[Dict[str, int]]  # None when zarr v3 sharding is off
    chunk_bytes: int
    shard_bytes: Optional[int]
    n_files: Optional[int]
    notes: List[str] = field(default_factory=list)
    # virtual grouping block to pass as ZarrPatchDataset(group_block=...) when
    # the store is not sharded; None when shards already play that role
    group_block: Optional[Dict[str, int]] = None

    def summary(self) -> str:
        lines = [f"chunks: {self.chunks}  (~{self.chunk_bytes / MiB:.1f} MiB)"]
        if self.shards is not None:
            lines.append(
                f"shards: {self.shards}  (~{self.shard_bytes / MiB:.0f} MiB, "
                f"~{self.n_files} data files)"
            )
        if self.group_block is not None:
            lines.append(f"group_block: {self.group_block}  "
                         "(pass to ZarrPatchDataset)")
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _round_up_multiple(value: int, base: int) -> int:
    return base * ceil(value / base)


def _min_group_edge(patch: int, chunk: int, size: int) -> int:
    """Smallest chunk multiple with <= 25% straddle overhead for this patch."""
    edge = max(chunk, _round_up_multiple(4 * (patch - 1), chunk))
    return min(edge, _round_up_multiple(size, chunk))


def recommend_layout(
    dim_sizes: Dict[str, int],
    patch_size: Dict[str, int],
    itemsize: int = 4,
    sampling: str = "random",
    stride: Optional[Dict[str, int]] = None,
    use_shards: bool = True,
    target_chunk_mib: float = 4.0,
    target_shard_mib: float = 256.0,
) -> LayoutRecommendation:
    """Recommend a zarr chunk (and shard) layout for a given training sampling.

    Parameters
    ----------
    dim_sizes: full cube dimension sizes, e.g. ``{"time": 1000, "lat": 2000, "lon": 4000}``.
    patch_size: patch length per sampled dim; other dims are read in full.
    itemsize: bytes per element of the (uncompressed) dtype.
    sampling: ``'random'`` or ``'grid'``.
    stride: stride per dim for grid sampling (defaults to the patch size).
    use_shards: emit a zarr v3 shard shape as well.
    """
    if sampling not in ("random", "grid"):
        raise ValueError("sampling must be 'random' or 'grid'")
    notes: List[str] = []
    chunks: Dict[str, int] = {}

    for d, n in dim_sizes.items():
        if d in patch_size:
            p = patch_size[d]
            if sampling == "grid":
                step = (stride or {}).get(d, p)
                # cover a whole patch and align to the sampling grid
                c = _round_up_multiple(max(p, step), step)
            else:
                c = p
            chunks[d] = min(c, n)
        else:
            chunks[d] = n  # full extent, trimmed below if the chunk is too big

    if sampling == "random":
        notes.append(
            "random sampling: chunk edge = patch edge along patched dims (the "
            "decode granularity). Reads must then be grouped by *several* "
            "chunks — a shard, or group_block — or each chunk is decoded up "
            "to 2^d times by neighbouring groups"
        )
    else:
        notes.append(
            "grid sampling: chunk edges aligned to the stride so patch reads "
            "fall on chunk boundaries"
        )

    # shrink un-patched dims (largest first) until the chunk fits the target
    target_chunk = target_chunk_mib * MiB
    free_dims = sorted(
        (d for d in dim_sizes if d not in patch_size),
        key=lambda d: -chunks[d],
    )
    for d in free_dims:
        size = prod(chunks.values()) * itemsize
        if size <= target_chunk:
            break
        shrink = ceil(size / target_chunk)
        chunks[d] = max(1, chunks[d] // shrink)
    chunk_bytes = prod(chunks.values()) * itemsize
    if chunk_bytes > 4 * target_chunk:
        notes.append(
            f"chunk is large (~{chunk_bytes / MiB:.0f} MiB) because the patch "
            "itself is large; consider whether all patched dims need full "
            "patch-sized chunks"
        )

    if not use_shards:
        n_chunks = prod(ceil(dim_sizes[d] / chunks[d]) for d in dim_sizes)
        group_block = {
            d: _min_group_edge(patch_size[d], chunks[d], dim_sizes[d])
            for d in patch_size
        }
        group_bytes = itemsize * prod(
            group_block.get(d, chunks[d]) for d in dim_sizes)
        return LayoutRecommendation(
            chunks, None, chunk_bytes, None, n_chunks,
            notes + [
                f"~{n_chunks} chunk files without sharding; if this is more "
                "than ~1e5, use zarr v3 sharding to spare the filesystem "
                "metadata servers",
                f"without shards, group reads with group_block={group_block} "
                f"(~{group_bytes / MiB:.0f} MiB per group read before the "
                "patch halo)",
            ],
            group_block=group_block,
        )

    # grow shard = chunk * k per dim, largest-gain first, until target size
    shards = dict(chunks)
    def shard_bytes() -> int:
        return prod(shards.values()) * itemsize

    target_shard = target_shard_mib * MiB
    grew = True
    while shard_bytes() < target_shard and grew:
        grew = False
        # double along the dim with the most remaining extent (in shards)
        candidates = [d for d in dim_sizes if shards[d] * 2 <= dim_sizes[d]
                      or shards[d] < dim_sizes[d]]
        if not candidates:
            break
        d = max(candidates, key=lambda d: dim_sizes[d] / shards[d])
        new = min(_round_up_multiple(shards[d] * 2, chunks[d]),
                  _round_up_multiple(dim_sizes[d], chunks[d]))
        if new != shards[d]:
            shards[d] = new
            grew = True

    short = {d: shards[d] for d in patch_size
             if shards[d] < _min_group_edge(patch_size[d], chunks[d],
                                            dim_sizes[d])}
    if short:
        notes.append(
            f"shard edges {short} are under ~4x the patch edge: groups will "
            "straddle neighbouring shards often; raise target_shard_mib or "
            "pass a larger group_block"
        )
    n_files = prod(ceil(dim_sizes[d] / shards[d]) for d in dim_sizes)
    notes.append(
        "sharding keeps the read unit (chunk) small while packing "
        f"~{prod(shards[d] // chunks[d] for d in dim_sizes)} chunks per file, "
        "reducing file count for HPC metadata servers"
    )
    return LayoutRecommendation(
        chunks, shards, chunk_bytes, shard_bytes(), n_files, notes
    )
