"""Compare zarrpatch with the loaders people would otherwise reach for.

Every loader reads the *same patch corners* from the *same store* and yields
``{"sst": tensor}`` batches through a torch ``DataLoader``:

- ``xarray isel``      map-style, ``xr.open_zarr(chunks=None)`` + ``isel`` per
                       patch — the "just use xarray" baseline.
- ``zarr per-patch``   zarrpatch with ``group_by_block=False`` (plain
                       zarr-python slicing per patch, no grouping).
- ``tensorstore``      map-style, one TensorStore read per patch, no cache.
- ``tensorstore+cache`` same with a 1 GiB chunk cache (``cache_pool``) per
                       process — the "a chunk cache solves random access" idea.
                       1 GiB holds this whole cube, so it is a best case;
                       ``tensorstore+cache 128M`` (~11% of the cube) is the
                       realistic "cube ≫ cache" regime.
- ``tensorstore batch`` ``__getitems__`` issues a whole batch of reads
                       concurrently (TensorStore's async strength), no cache.
                       TensorStore decodes on its own thread pool (all cores by
                       default), so ``tensorstore batch 1t`` also runs it with
                       ``data_copy_concurrency=1`` for a per-core comparison.
- ``xbatcher``         ``BatchGenerator`` + ``MapDataset`` (grid scenario only:
                       xbatcher enumerates a regular tiling, it has no random
                       corners), shuffled by the DataLoader.
- ``zarrpatch``        block-grouped reads + shuffle buffer (``"auto"``);
                       on the unsharded store also with ``group_block``.
- ``zarrpatch on TS``  the same planner/grouping/shuffle with
                       ``TensorStoreReader`` (``backend="tensorstore"``, one
                       decode thread) — separates "grouping" from
                       "zarr-python as the decode engine".

Scenarios: ``random`` (N uniform corners over the whole cube) and ``grid``
(a shuffled non-overlapping tiling of the whole cube — xbatcher's home turf).

Each configuration builds one dataset and one DataLoader (``persistent_workers``
when multi-worker, as in training) and runs 3 epochs; ``set_epoch`` switches
every loader to a fresh corner draw (random scenario: zarrpatch's
``RandomSampler`` and the baselines draw *identical* corners per epoch). Epoch
1 runs after a best-effort page-cache eviction (``posix_fadvise(DONTNEED)`` on
every store file — on GPFS the pagepool may still hold data, so it is not a
guaranteed cold read) and includes worker start-up; epochs 2–3 are steady
state, and the best of them is reported as "warm".

Wall-clock rates alone favour loaders that use more cores, so every pass also
records **CPU seconds** summed over the main process (all threads) and every
DataLoader worker (polled with psutil while they live), and reports patches
per CPU-second — the number that matters when CPU cores per GPU are the
budget. On a shared node, wall times are noisy; CPU time much less so.

Usage (bench env has xbatcher + tensorstore)::

    OMP_NUM_THREADS=1 pixi run -e ecosystem python benchmarks/ecosystem_bench.py \\
        [--root benchmarks/data/eco] [--patches 20000] [--workers 0 8]

Writes ``benchmarks/results_ecosystem.md`` and ``.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from _env import env_path
from zarrpatch import ListSampler, RandomSampler, TensorStoreReader, ZarrPatchDataset

DIMS = {"time": 200, "lat": 1200, "lon": 1200}  # ~1.1 GiB float32
PATCH = {"time": 5, "lat": 64, "lon": 64}
CHUNKS = {"time": 10, "lat": 100, "lon": 100}
SHARDS = {"time": 50, "lat": 300, "lon": 300}
GROUP_BLOCK = {"time": 20, "lat": 300, "lon": 300}  # >= ~4 x (patch - 1)
BATCH = 32
TS_CACHE = 1 << 30


# -- stores -------------------------------------------------------------------

def write_stores(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    stores = {"sharded": root / "cube_sharded.zarr",
              "unsharded": root / "cube_unsharded.zarr"}
    if all(p.exists() for p in stores.values()):
        return {k: str(v) for k, v in stores.items()}
    rng = np.random.default_rng(0)
    data = rng.normal(size=tuple(DIMS.values())).astype("float32")
    data = np.cumsum(data, axis=-1) / 10  # smooth-ish, compressible
    ds = xr.Dataset({"sst": (list(DIMS), data)})
    chunks = tuple(CHUNKS.values())
    ds.to_zarr(stores["unsharded"], zarr_format=3, mode="w",
               encoding={"sst": {"chunks": chunks}})
    ds.to_zarr(stores["sharded"], zarr_format=3, mode="w",
               encoding={"sst": {"chunks": chunks,
                                 "shards": tuple(SHARDS.values())}})
    return {k: str(v) for k, v in stores.items()}


def evict(path: str) -> None:
    """Best-effort page-cache eviction (no root needed; GPFS may ignore it)."""
    for dirpath, _, files in os.walk(path):
        for f in files:
            fd = os.open(os.path.join(dirpath, f), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


# -- corners ------------------------------------------------------------------

def random_corners(n: int, epoch: int) -> list:
    return RandomSampler(n).sample(DIMS, PATCH, np.random.default_rng((0, epoch)))


def grid_corners() -> list:
    axes = {"time": range(0, DIMS["time"] - PATCH["time"] + 1, PATCH["time"]),
            "lat": range(0, DIMS["lat"] - PATCH["lat"] + 1, PATCH["lat"]),
            "lon": range(0, DIMS["lon"] - PATCH["lon"] + 1, PATCH["lon"])}
    return [{"time": t, "lat": la, "lon": lo}
            for t in axes["time"] for la in axes["lat"] for lo in axes["lon"]]


def _sl(c, d, offset=0):
    return slice(c[d] + offset, c[d] + offset + PATCH[d])


# -- baseline datasets (map-style, handles opened lazily per worker) -----------

class EpochCorners(Dataset):
    """Map-style base: the corner list follows a shared (cross-process) epoch,
    so persistent workers see ``set_epoch`` like ``ZarrPatchDataset`` does."""

    def __init__(self, scenario, n):
        self.scenario, self.n = scenario, n
        self.epoch = mp.get_context("spawn").Value("l", 0)
        self._cache = {}

    def set_epoch(self, epoch):
        self.epoch.value = epoch

    def corners(self):
        e = self.epoch.value
        if e not in self._cache:
            self._cache = {e: random_corners(self.n, e) if self.scenario == "random"
                           else grid_corners()}
        return self._cache[e]

    def __len__(self):
        return self.n if self.scenario == "random" else len(grid_corners())


class XarrayPatches(EpochCorners):
    def __init__(self, path, scenario, n):
        super().__init__(scenario, n)
        self.path = path
        self._da = None

    def __getitem__(self, i):
        if self._da is None:
            self._da = xr.open_zarr(self.path, chunks=None)["sst"]
        c = self.corners()[i]
        return {"sst": self._da.isel(time=_sl(c, "time"), lat=_sl(c, "lat"),
                                     lon=_sl(c, "lon")).values}

    def __getstate__(self):
        return {**self.__dict__, "_da": None}


class TensorStorePatches(EpochCorners):
    def __init__(self, path, scenario, n, cache_bytes=0, batched=False,
                 threads=None):
        super().__init__(scenario, n)
        self.path = path
        self.cache_bytes, self.batched, self.threads = cache_bytes, batched, threads
        self._arr = None

    def _open(self):
        import tensorstore as ts

        if self._arr is None:
            spec = {"cache_pool": {"total_bytes_limit": self.cache_bytes}}
            if self.threads:
                spec["data_copy_concurrency"] = {"limit": self.threads}
            ctx = ts.Context(spec)
            self._arr = ts.open({
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": self.path + "/sst"},
            }, read=True, context=ctx).result()
        return self._arr

    def _view(self, c):
        return self._open()[_sl(c, "time"), _sl(c, "lat"), _sl(c, "lon")]

    def __getitem__(self, i):
        return {"sst": self._view(self.corners()[i]).read().result()}

    def __getitems__(self, indices):
        if not self.batched:
            return [self[i] for i in indices]
        corners = self.corners()
        futures = [self._view(corners[i]).read() for i in indices]
        return [{"sst": f.result()} for f in futures]

    def __getstate__(self):
        return {**self.__dict__, "_arr": None}


class XbatcherPatches(Dataset):
    """xbatcher's own torch adapter over a regular tiling."""

    def __init__(self, path):
        self.path = path
        self._ds = None

    def set_epoch(self, epoch):  # fixed tiling; the DataLoader reshuffles
        pass

    def _open(self):
        if self._ds is None:
            import xbatcher
            from xbatcher.loaders.torch import MapDataset

            da = xr.open_zarr(self.path, chunks=None)["sst"]
            gen = xbatcher.BatchGenerator(da, input_dims=PATCH)
            self._ds = MapDataset(gen)
        return self._ds

    def __len__(self):
        return (DIMS["time"] // PATCH["time"]) * (DIMS["lat"] // PATCH["lat"]) * (DIMS["lon"] // PATCH["lon"])

    def __getitem__(self, i):
        x = self._open()[i]
        return {"sst": x[0] if isinstance(x, tuple) else x}

    def __getstate__(self):
        return {**self.__dict__, "_ds": None}


# -- runner -------------------------------------------------------------------

def loader_for(ds, workers, shuffle=False):
    kw = dict(batch_size=BATCH, num_workers=workers)
    if workers:
        # timeout: a batch that takes > 10 min is a hang (seen once) —
        # fail loudly instead of stalling the run
        kw.update(multiprocessing_context="forkserver", persistent_workers=True,
                  timeout=600)
    if not isinstance(ds, torch.utils.data.IterableDataset):
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


class CpuMeter:
    """CPU seconds of this process (all threads) plus its descendants.

    Worker processes are polled every 50 ms while alive, so up to ~50 ms of a
    worker's final CPU time can be missed (<2% at these run lengths).
    """

    def __init__(self):
        import psutil

        self.proc = psutil.Process()
        self.seen, self.base = {}, {}

    @staticmethod
    def _cpu(p) -> float:
        t = p.cpu_times()
        return t.user + t.system

    def _poll(self):
        for c in self.proc.children(recursive=True):
            try:
                self.seen[c.pid] = self._cpu(c)
            except Exception:  # exited between listing and reading
                pass

    def __enter__(self):
        self._poll()
        self.base = dict(self.seen)
        self.main0 = self._cpu(self.proc)
        self._stop = threading.Event()

        def loop():
            while not self._stop.wait(0.05):
                self._poll()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._poll()
        self._stop.set()
        self._thread.join()
        children = sum(v - self.base.get(pid, 0.0) for pid, v in self.seen.items())
        self.cpu_s = self._cpu(self.proc) - self.main0 + children


def time_loader(loader) -> dict:
    with CpuMeter() as meter:
        t0 = time.perf_counter()
        first, n, n_first = None, 0, 0
        for batch in loader:
            k = len(batch["sst"])
            n += k
            if first is None:
                first, n_first = time.perf_counter() - t0, k
        wall = time.perf_counter() - t0
    steady = (n - n_first) / (wall - first) if wall > first else float("nan")
    return {"patches": n, "wall_s": wall, "first_batch_s": first,
            "patches_per_s": n / wall, "steady_patches_per_s": steady,
            "cpu_s": meter.cpu_s, "patches_per_cpu_s": n / meter.cpu_s}


def configs(scenario: str, stores: dict, n: int):
    """Yield (name, store_key, make_dataset(), shuffle)."""
    def zp(path, ts_threads=0, **kw):
        def make():
            sampler = (RandomSampler(n) if scenario == "random"
                       else ListSampler(grid_corners()))
            src = TensorStoreReader(path, threads=ts_threads) if ts_threads else path
            return ZarrPatchDataset(src, PATCH, sampler=sampler,
                                    shuffle_buffer="auto", seed=0, **kw)
        return make

    grid = scenario == "grid"
    for key in ("unsharded", "sharded"):
        path = stores[key]
        yield ("xarray isel", key,
               lambda p=path: XarrayPatches(p, scenario, n), grid)
        yield ("zarr per-patch", key,
               zp(path, group_by_block=False, prefetch=0, shuffle=grid), False)
        yield ("tensorstore", key,
               lambda p=path: TensorStorePatches(p, scenario, n), grid)
        yield ("tensorstore+cache", key,
               lambda p=path: TensorStorePatches(p, scenario, n, TS_CACHE), grid)
        yield ("tensorstore+cache 128M", key,
               lambda p=path: TensorStorePatches(p, scenario, n, 128 << 20), grid)
        yield ("tensorstore batch", key,
               lambda p=path: TensorStorePatches(p, scenario, n, 0, True), grid)
        yield ("tensorstore batch 1t", key,
               lambda p=path: TensorStorePatches(p, scenario, n, 0, True, 1), grid)
        if grid:
            yield ("xbatcher", key, lambda p=path: XbatcherPatches(p), True)
        yield ("zarrpatch", key, zp(path), False)
        if key == "unsharded":
            yield ("zarrpatch group_block", key, zp(path, group_block=GROUP_BLOCK),
                   False)
        gb = {"group_block": GROUP_BLOCK} if key == "unsharded" else {}
        yield ("zarrpatch on TS 1t", key, zp(path, ts_threads=1, **gb), False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(
        env_path("ZARRPATCH_BENCH_DATA", "benchmarks/data") / "eco"))
    ap.add_argument("--patches", type=int, default=20000)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--scenarios", nargs="+", default=["random", "grid"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter on loader names")
    ap.add_argument("--out", default="benchmarks/results_ecosystem")
    args = ap.parse_args()

    import warnings
    warnings.simplefilter("ignore", UserWarning)  # straddle warning is expected
    stores = write_stores(Path(args.root))
    rows = []
    for scenario in args.scenarios:
        for workers in args.workers:
            for name, key, make, shuffle in configs(scenario, stores, args.patches):
                if args.only and not any(s in name for s in args.only):
                    continue
                ds = make()
                loader = loader_for(ds, workers, shuffle)
                passes = []
                for epoch in range(3):
                    ds.set_epoch(epoch)
                    if epoch == 0:
                        evict(stores[key])
                    passes.append(time_loader(loader))
                del loader, ds  # shut persistent workers down
                warm = max(passes[1:], key=lambda r: r["patches_per_s"])
                row = {"scenario": scenario, "store": key, "loader": name,
                       "workers": workers, "first": passes[0], "warm": warm}
                rows.append(row)
                print(f"{scenario:6s} {key:9s} w={workers} {name:22s} "
                      f"epoch1 {passes[0]['patches_per_s']:8.1f}/s "
                      f"(first batch {passes[0]['first_batch_s']:.1f}s)  "
                      f"warm {warm['patches_per_s']:8.1f}/s "
                      f"cpu {warm['cpu_s']:6.1f}s "
                      f"({warm['patches_per_cpu_s']:7.1f}/cpu-s)", flush=True)
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps({
        "host": platform.node(), "cpus": len(os.sched_getaffinity(0)),
        "loadavg": os.getloadavg(), "dims": DIMS, "patch": PATCH, "chunks": CHUNKS,
        "shards": SHARDS, "group_block": GROUP_BLOCK, "batch": BATCH,
        "patches": args.patches, "rows": rows}, indent=1))
    # end-to-end patches/s is the headline: "steady" rates exclude the first
    # batch and so flatter loaders that front-load work (a shuffle buffer
    # filling, worker start-up); they stay in the JSON only
    lines = ["| scenario | store | loader | workers | epoch 1 /s "
             "| epoch 1 first batch [s] | warm /s | warm CPU [s] | warm /CPU-s |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['store']} | {r['loader']} | {r['workers']} "
            f"| {r['first']['patches_per_s']:.0f} "
            f"| {r['first']['first_batch_s']:.2f} "
            f"| {r['warm']['patches_per_s']:.0f} "
            f"| {r['warm']['cpu_s']:.1f} | {r['warm']['patches_per_cpu_s']:.0f} |")
    out.with_suffix(".table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
