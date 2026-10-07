"""Benchmark zarrpatch loading strategies against baseline formats.

Generates a synthetic geophysical datacube, writes it as netCDF, zarr v2,
zarr v3 (unsharded) and zarr v3 (sharded), then measures random-patch loading
throughput for:

- naive per-patch reads on every format (the classic xbatcher-style baseline)
- block-grouped reads (zarr v3, without shuffle buffer)
- block-grouped reads + shuffle buffer (the full zarrpatch pipeline)

Usage:  pixi run bench  [--patches 2000] [--workers 0] [--size small|medium]
Writes results to benchmarks/results.md.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from _env import env_path
from zarrpatch import (
    ListSampler,
    RandomSampler,
    ZarrPatchDataset,
    benchmark,
    print_results,
    recommend_layout,
    results_to_markdown,
)

SIZES = {
    "small": {"time": 60, "lat": 360, "lon": 360},   # ~30 MiB float32
    "medium": {"time": 100, "lat": 600, "lon": 600},  # ~137 MiB float32
    "large": {"time": 200, "lat": 1200, "lon": 1200},  # ~1.1 GiB float32
}
PATCH = {"time": 5, "lat": 64, "lon": 64}
CHUNKS = {"time": 10, "lat": 100, "lon": 100}
SHARDS = {"time": 50, "lat": 300, "lon": 300}


def make_cube(dims: dict) -> xr.Dataset:
    rng = np.random.default_rng(0)
    data = rng.normal(size=tuple(dims.values())).astype("float32")
    # make it compressible like smooth geophysical fields
    data = np.cumsum(data, axis=-1) / 10
    return xr.Dataset({"sst": (list(dims), data)})


def write_stores(ds: xr.Dataset, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    chunk_t = tuple(CHUNKS[d] for d in ds.sst.dims)
    shard_t = tuple(SHARDS[d] for d in ds.sst.dims)
    stores = {}

    p = root / "cube.nc"
    if not p.exists():
        ds.to_netcdf(p, encoding={"sst": {"chunksizes": chunk_t, "zlib": True,
                                          "complevel": 1}})
    stores["netcdf"] = p

    specs = {
        "zarr_v2": dict(zarr_format=2, encoding={"sst": {"chunks": chunk_t}}),
        "zarr_v3": dict(zarr_format=3, encoding={"sst": {"chunks": chunk_t}}),
        "zarr_v3_sharded": dict(
            zarr_format=3,
            encoding={"sst": {"chunks": chunk_t, "shards": shard_t}},
        ),
    }
    for name, kw in specs.items():
        p = root / f"cube_{name}.zarr"
        if not p.exists():
            ds.to_zarr(p, mode="w", **kw)
        stores[name] = p
    return stores


def file_count(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.is_file()) if path.is_dir() else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--patches", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--buffer", type=int, default=2048)
    ap.add_argument("--data-dir", type=Path,
                    default=env_path("ZARRPATCH_BENCH_DATA",
                                     Path(__file__).parent / "data"))
    ap.add_argument("--fresh", action="store_true",
                    help="regenerate the stores from scratch")
    args = ap.parse_args()

    dims = SIZES[args.size]
    data_dir = args.data_dir / args.size
    if args.fresh and data_dir.exists():
        shutil.rmtree(data_dir)

    print(f"cube dims {dims}, patch {PATCH}, {args.patches} random patches, "
          f"batch {args.batch_size}, workers {args.workers}")
    print("\nrecommended layout for this setup:")
    print(recommend_layout(dims, PATCH, target_shard_mib=64).summary())

    print("\nwriting stores (skipped if present)...")
    stores = write_stores(make_cube(dims), data_dir)
    for name, p in stores.items():
        print(f"  {name:16s} {file_count(p):6d} files  {p}")

    # identical corners everywhere so every loader does the same work
    rng = np.random.default_rng(123)
    corners = RandomSampler(args.patches).sample(dims, PATCH, rng)
    sampler = ListSampler(corners)

    def loader(path, *, grouped, buffer, prefetch=0, backend=None, shuffle=True):
        ds = ZarrPatchDataset(
            str(path), PATCH, sampler=sampler, backend=backend,
            group_by_block=grouped, shuffle=shuffle,
            shuffle_buffer=buffer, prefetch=prefetch, seed=0,
        )
        return lambda: DataLoader(ds, batch_size=args.batch_size,
                                  num_workers=args.workers)

    configs = [
        ("netcdf / naive", loader(stores["netcdf"], grouped=False, buffer=0)),
        ("zarr v2 / naive", loader(stores["zarr_v2"], grouped=False, buffer=0)),
        ("zarr v3 / naive", loader(stores["zarr_v3"], grouped=False, buffer=0)),
        ("zarr v3 sharded / naive",
         loader(stores["zarr_v3_sharded"], grouped=False, buffer=0)),
        ("zarr v3 / grouped, no buffer",
         loader(stores["zarr_v3"], grouped=True, buffer=0)),
        ("zarr v3 sharded / grouped, no buffer",
         loader(stores["zarr_v3_sharded"], grouped=True, buffer=0)),
        ("zarr v3 sharded / grouped + buffer",
         loader(stores["zarr_v3_sharded"], grouped=True, buffer=args.buffer)),
        ("zarr v3 sharded / grouped + buffer + prefetch",
         loader(stores["zarr_v3_sharded"], grouped=True, buffer=args.buffer,
                prefetch=1)),
    ]

    print("\nrunning (best of 2 passes each, page cache warm for all)...")
    results = []
    for name, make_iter in configs:
        res = benchmark(make_iter, name=name, repeats=2)
        # report patches rather than batches
        res = type(res)(name, args.patches, res.wall_s,
                        res.first_item_s, res.bytes_total)
        print(f"  {name:40s} {res.items_per_s:8.0f} patches/s")
        results.append(res)

    print()
    print_results(results, title=f"random-patch loading — {args.size} cube",
                  baseline="netcdf / naive")

    out = Path(__file__).parent / "results.md"
    header = (
        f"# zarrpatch benchmark — {args.size} cube\n\n"
        f"dims {dims}, patch {PATCH}, {args.patches} random patches, "
        f"batch {args.batch_size}, workers {args.workers}, "
        f"buffer {args.buffer}\n\n"
        f"chunks {CHUNKS}, shards {SHARDS}. `items` counts patches; speedup "
        f"is vs `netcdf / naive`.\n\n"
    )
    out.write_text(header + results_to_markdown(results, baseline="netcdf / naive") + "\n")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
