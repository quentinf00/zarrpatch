"""Where does a zarrpatch epoch spend its time?

Splits one warm epoch of random-patch loading into its stages, in a single
process with read-ahead disabled so the stages do not overlap:

- **open**: build the dataset (open the store, read metadata);
- **plan**: draw corners and group them by block, up to the first read;
- **read**: region reads (filesystem + decompression), with the decoded bytes
  compared to the bytes the patches actually need (read amplification);
- **decode**: the part of *read* that is decompression, estimated as the read
  time on the compressed store minus the same reads on an uncompressed copy;
- **slice + shuffle**: cutting patches out of decoded blocks and the shuffle
  buffer (the rest of the iteration time);
- **collate**: ``default_collate`` into batches.

A second table shows what ``DataLoader`` workers add: the same pipeline with
0/4/8 workers next to a control dataset that returns in-memory patches, which
is the ceiling any reader can reach through workers (collate + transfer to the
main process).

Usage:  pixi run python benchmarks/time_breakdown.py [--patches 4000]
Writes benchmarks/results_breakdown.md. Stores go under
``$ZARRPATCH_BENCH_DATA`` (default ``benchmarks/data``), see ``.env.example``.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from _env import env_path
from run_bench import CHUNKS, PATCH, SHARDS, SIZES, make_cube
from zarrpatch import RandomSampler, ZarrPatchDataset

BATCH = 32


def write_stores(root: Path, size: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    ds = None
    chunks = tuple(CHUNKS.values())
    specs = {
        "zstd": {"chunks": chunks},
        "raw": {"chunks": chunks, "compressors": None},
        "zstd_sharded": {"chunks": chunks, "shards": tuple(SHARDS.values())},
    }
    stores = {}
    for name, enc in specs.items():
        p = root / f"breakdown_{size}_{name}.zarr"
        if not p.exists():
            ds = make_cube(SIZES[size]) if ds is None else ds
            ds.to_zarr(p, mode="w", zarr_format=3, encoding={"sst": enc})
        stores[name] = p
    return stores


def make_dataset(path, n, **kw):
    return ZarrPatchDataset(str(path), PATCH, sampler=RandomSampler(n),
                            prefetch=0, seed=0, **kw)


def stage_times(path, n, **kw) -> dict:
    t0 = time.perf_counter()
    ds = make_dataset(path, n, **kw)
    t_open = time.perf_counter() - t0
    list(ds)                                   # warm-up epoch (page cache)
    ds.set_epoch(1)

    reads = {"s": 0.0, "bytes": 0, "groups": 0, "first": None}
    inner = ds._read_group

    def timed(group):
        t = time.perf_counter()
        reads["first"] = reads["first"] or t
        out = inner(group)
        reads["s"] += time.perf_counter() - t
        reads["bytes"] += sum(a.nbytes for a in out[2].values())
        reads["groups"] += 1
        return out

    ds._read_group = timed
    t0 = time.perf_counter()
    first_patch = None
    patches = []
    for p in ds:
        first_patch = first_patch or time.perf_counter()
        patches.append(p)
    total = time.perf_counter() - t0

    default_collate(patches[:BATCH])           # warm-up
    t = time.perf_counter()
    for i in range(0, len(patches) - BATCH + 1, BATCH):
        default_collate(patches[i:i + BATCH])
    t_collate = time.perf_counter() - t

    patch_bytes = sum(a.nbytes for a in patches[0].values()) * len(patches)
    t_plan = reads["first"] - t0
    return {
        "open": t_open, "plan": t_plan, "read": reads["s"],
        "slice_shuffle": total - t_plan - reads["s"], "collate": t_collate,
        "ttfp": first_patch - t0, "patches": len(patches),
        "groups": reads["groups"], "amplification": reads["bytes"] / patch_bytes,
    }


class InMemory(Dataset):
    """Control: patches already in memory, so only collate + transfer remain."""

    def __init__(self, n):
        rng = np.random.default_rng(0)
        self.pool = [{"sst": rng.normal(size=tuple(PATCH.values())).astype("f4")}
                     for _ in range(256)]
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v.copy() for k, v in self.pool[i % len(self.pool)].items()}


def loader_rate(ds, workers, epochs=2) -> float:
    kw = dict(batch_size=BATCH, num_workers=workers)
    if workers:
        kw.update(multiprocessing_context="forkserver", persistent_workers=True)
    loader = DataLoader(ds, **kw)
    rate = 0.0
    for epoch in range(epochs):                  # keep the warm epoch
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        t = time.perf_counter()
        n = sum(len(b["sst"]) for b in loader)
        rate = n / (time.perf_counter() - t)
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--patches", type=int, default=4000)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "results_breakdown.md")
    args = ap.parse_args()
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    stores = write_stores(env_path("ZARRPATCH_BENCH_DATA",
                                   Path(__file__).parent / "data") / "breakdown",
                          args.size)
    n = args.patches
    group = {d: 3 * c for d, c in CHUNKS.items()}
    configs = {
        "chunk groups, zstd": (stores["zstd"], {}),
        "chunk groups, uncompressed": (stores["raw"], {}),
        "shard groups, zstd": (stores["zstd_sharded"], {}),
        "group_block 3x chunk, zstd": (stores["zstd"], {"group_block": group}),
    }
    rows = {}
    for name, (path, kw) in configs.items():
        rows[name] = stage_times(path, n, **kw)
        r = rows[name]
        print(f"{name:28s} read {r['read']:.2f}s  slice+shuffle "
              f"{r['slice_shuffle']:.2f}s  collate {r['collate']:.2f}s", flush=True)

    decode = {k: rows[k]["read"] - rows["chunk groups, uncompressed"]["read"]
              for k in ("chunk groups, zstd",)}

    rates = {}
    for w in args.workers:
        rates[w] = (loader_rate(make_dataset(stores["zstd_sharded"], n), w),
                    loader_rate(InMemory(n), w))
        print(f"workers={w}: zarrpatch {rates[w][0]:.0f}/s, "
              f"in-memory control {rates[w][1]:.0f}/s", flush=True)

    patch_mib = np.prod(list(PATCH.values())) * 4 / 2**20
    L = [
        "# Where the time goes: one warm epoch, stage by stage\n",
        f"Synthetic `{args.size}` cube {SIZES[args.size]} float32, zarr v3, "
        f"chunks {CHUNKS}, shards {SHARDS}; {n} random {PATCH} patches "
        f"({patch_mib:.2f} MiB each), batch {BATCH}. One process, read-ahead off "
        f"(`prefetch=0`), warm page cache, `torch.set_num_threads(1)`. "
        f"{os.cpu_count()} CPUs, Python {platform.python_version()}, "
        f"torch {torch.__version__}. Produced by `benchmarks/time_breakdown.py`.\n",
        "## Stages (seconds per epoch; % of the epoch excluding open)\n",
        "| configuration | open | plan | read (I/O + decode + zarr) | slice + shuffle "
        "| collate | first patch after | groups | read amplification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        tot = r["plan"] + r["read"] + r["slice_shuffle"] + r["collate"]
        pct = lambda x: f"{x:.2f} ({100 * x / tot:.0f}%)"  # noqa: E731
        L.append(f"| {name} | {r['open']:.2f} | {pct(r['plan'])} | {pct(r['read'])} "
                 f"| {pct(r['slice_shuffle'])} | {pct(r['collate'])} "
                 f"| {r['ttfp']:.2f} | {r['groups']} | {r['amplification']:.1f}× |")
    d = decode["chunk groups, zstd"]
    L += [
        "",
        f"Decompression (zstd minus uncompressed reads, chunk groups): "
        f"**{d:.2f} s**, {100 * d / max(rows['chunk groups, zstd']['read'], 1e-9):.0f}% "
        "of the read stage.\n",
        "## What DataLoader workers add (samples/s, warm epoch)\n",
        "| workers | zarrpatch (shard groups) | in-memory control (collate + transfer ceiling) |",
        "|---:|---:|---:|",
    ]
    L += [f"| {w} | {a:,.0f} | {b:,.0f} |" for w, (a, b) in rates.items()]
    args.out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
