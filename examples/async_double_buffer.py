"""Batteries-included example: async double-buffered materialize for training.

The pattern this showcases
--------------------------
A ``time × lat × lon`` cube with *millions* of possible patches is too big to
freeze whole, and drawing a fresh random corner set every single epoch and
reading it live is slow. So instead:

1. **Materialize a pool.** Draw ``pool_size`` random patches with
   ``RandomSampler`` and freeze them to an *uncompressed* HDF5 file with
   ``materialize()`` (one sequential pass — the block-grouped reader's best
   access pattern).
2. **Train several reshuffled epochs on that pool.** Read it back with
   ``HDF5PatchDataset`` + ``DataLoader(shuffle=True)``: every epoch reorders the
   same frozen patches for free (~37× faster than re-reading the cube). Training
   on one pool for a few epochs is fine — reshuffling keeps the gradient noise
   fresh; the pool is just a *sample* of the millions of possible patches.
3. **Meanwhile, build the next pool in the background.** A single worker process
   materializes a brand-new random pool (a different HDF5 file) while the GPU
   trains on the current one.
4. **Swap when it's ready.** As soon as the background build finishes, switch the
   training loader to the new pool and kick off the following build. The two
   pools ping-pong between two files (``pool_A.h5`` / ``pool_B.h5``), so the
   trainer and the builder never touch the same file.
5. **Validation** is a fixed corner set frozen once to an uncompressed HDF5 file
   and reused every epoch.

Why the primitives suffice
--------------------------
- fresh random patches per build ← ``RandomSampler(pool_size)`` +
  ``set_epoch(k)`` (each build uses a distinct epoch, so it draws a new corner
  set from ``rng((seed, epoch))``);
- the freeze ← ``materialize(ds, path, overwrite=True)`` (single writer);
- reshuffled reads ← ``HDF5PatchDataset(path)`` + ``DataLoader(shuffle=True)``;
- the async overlap ← a stock ``ProcessPoolExecutor(max_workers=1)``. The child
  rebuilds its *own* dataset from the cube **path** (fresh reader + fresh h5
  handle), sidestepping h5py/zarr fork-safety entirely. No library feature owns
  the training loop — this orchestration is deliberately application code.

Caveats worth designing around
------------------------------
- **Build-time vs epoch-time.** If one background build takes longer than one
  training epoch, you simply train *more* reshuffled epochs on the current pool
  before the swap — which is exactly what the loop below does (it only swaps when
  ``future.done()``). If builds are *much* slower than epochs you are pool-bound;
  either enlarge the pool, shard the build across processes into several files
  read via ``ConcatDataset``, or just read live.
- **Disk.** Two training pools + val live on disk at once. Per pool that is
  ``pool_size × bytes_per_patch`` — here a raw patch is
  ``time·lat·lon·4`` bytes. At the demo defaults that is a few hundred MB; at a
  realistic ``pool_size=64_000`` with big patches it is hundreds of GB. Size it.
- **Cold start.** The first pool has to be built synchronously before training
  can begin; overlap kicks in from the second pool onward.

Usage
-----
    pixi run python examples/async_double_buffer.py            # quick demo
    pixi run python examples/async_double_buffer.py \
        --pool-size 64000 --batches-per-epoch 1000 --epochs 40 # realistic scale

Everything is written under a scratch/tmp dir and cleaned up on exit.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zarrpatch import (
    HDF5PatchDataset,
    RandomSampler,
    ZarrPatchDataset,
    materialize,
)

VARIABLE = "sea_surface_height"
PATCH_SIZE: Dict[str, int] = {"time": 4, "lat": 64, "lon": 64}


# --------------------------------------------------------------------------- #
# synthetic cube
# --------------------------------------------------------------------------- #
def build_synthetic_cube(path: Path, sizes: Dict[str, int]) -> None:
    """Write a smooth, compressible ``time × lat × lon`` zarr cube."""
    import xarray as xr

    rng = np.random.default_rng(0)
    data = rng.normal(size=tuple(sizes.values())).astype("float32")
    data = np.cumsum(data, axis=-1) / 10.0  # smear like a geophysical field
    ds = xr.Dataset({VARIABLE: (list(sizes), data)})
    chunks = {"time": 8, "lat": 128, "lon": 128}
    ds.to_zarr(
        path, mode="w", zarr_format=3,
        encoding={VARIABLE: {"chunks": tuple(chunks[d] for d in sizes)}},
    )


def n_possible_patches(sizes: Dict[str, int]) -> int:
    n = 1
    for d, p in PATCH_SIZE.items():
        n *= sizes[d] - p + 1
    return n


# --------------------------------------------------------------------------- #
# read-time transform: raw patch dict -> a single float32 tensor
# --------------------------------------------------------------------------- #
def to_tensor(patch: Dict[str, np.ndarray]) -> torch.Tensor:
    """``{var: (T, H, W), "corner": ...}`` -> ``(T, H, W)`` float32 tensor.

    Applied on *read* (``HDF5PatchDataset(transform=...)``), not baked into the
    frozen file — so the pool stores raw patches and you keep the freedom to
    change/augment the transform between epochs. Freeze the post-transform sample
    instead (attach the transform to the dataset you materialize) when you want
    the very fastest reads and no per-read compute.
    """
    return torch.from_numpy(np.asarray(patch[VARIABLE])).float()


# --------------------------------------------------------------------------- #
# background pool builder (runs in its own process)
# --------------------------------------------------------------------------- #
def build_pool(
    cube_path: str,
    variables: List[str],
    pool_size: int,
    seed: int,
    epoch: int,
    out_path: str,
) -> str:
    """Materialize ``pool_size`` fresh random patches to ``out_path``.

    Top-level (picklable) so it can run in a ``ProcessPoolExecutor`` worker. It
    opens the cube itself from ``cube_path`` — nothing stateful is shipped across
    the process boundary, so h5py/zarr handles are never shared across a fork.
    ``epoch`` selects the RNG stream, so each build draws a different corner set.
    """
    ds = ZarrPatchDataset(
        cube_path, PATCH_SIZE, variables=variables,
        sampler=RandomSampler(pool_size),
        group_by_block=True, shuffle=False, prefetch=1, seed=seed,
    )
    ds.set_epoch(epoch)  # fresh random corners for this build
    materialize(ds, out_path, compression=None, overwrite=True)
    return out_path


# --------------------------------------------------------------------------- #
# a tiny model, just to give the GPU/CPU something to chew on
# --------------------------------------------------------------------------- #
class TinyAutoencoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(loader, model, opt, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x in loader:
        x = x.to(device)
        opt.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, x)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def validate(loader, model, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x in loader:
        x = x.to(device)
        loss = nn.functional.mse_loss(model(x), x)
        total += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-size", type=int, default=4096,
                    help="patches materialized per training pool "
                         "(realistic: 64000)")
    ap.add_argument("--batches-per-epoch", type=int, default=0,
                    help="0 = use the whole pool each epoch (pool_size/batch "
                         "steps); realistic: 1000")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--workdir", type=str, default="",
                    help="where the cube + pools live (default: a tmp dir)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    owns_workdir = not args.workdir
    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="zarrpatch_async_"))
    workdir.mkdir(parents=True, exist_ok=True)
    cube_path = workdir / "cube.zarr"
    pool_paths = [str(workdir / "pool_A.h5"), str(workdir / "pool_B.h5")]
    val_path = str(workdir / "val.h5")

    # ---- one-time setup: cube + fixed validation snapshot ----
    sizes = {"time": 48, "lat": 384, "lon": 384}
    print(f"workdir: {workdir}  device: {device}")
    if not cube_path.exists():
        build_synthetic_cube(cube_path, sizes)
    print(f"cube {dict(sizes)}  →  {n_possible_patches(sizes):,} possible "
          f"{PATCH_SIZE} patches")

    variables = [VARIABLE]
    val_ds = ZarrPatchDataset(
        str(cube_path), PATCH_SIZE, variables=variables,
        sampler=RandomSampler(args.val_size),
        group_by_block=True, shuffle=False, prefetch=1, seed=12345,
    )
    t = time.perf_counter()
    materialize(val_ds, val_path, compression=None, overwrite=True)
    print(f"val: froze {args.val_size} patches (uncompressed) in "
          f"{time.perf_counter() - t:.1f}s")
    val_loader = DataLoader(
        HDF5PatchDataset(val_path, transform=to_tensor),
        batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
    )

    model = TinyAutoencoder(PATCH_SIZE["time"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- cold start: build the first pool synchronously ----
    print("cold-start build of the first pool …")
    t = time.perf_counter()
    build_pool(str(cube_path), variables, args.pool_size, 0, 0, pool_paths[0])
    cold_s = time.perf_counter() - t
    print(f"  first pool ready in {cold_s:.1f}s")

    # spawn the builder pool and kick off the SECOND pool in the background
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool_exec:
        cur = 0            # index of the pool the trainer reads
        build_seq = 1      # epoch id passed to build_pool (drives fresh draws)
        # the background build writes the OTHER file (cur ^ 1)
        future: Future = pool_exec.submit(
            build_pool, str(cube_path), variables, args.pool_size, 0,
            build_seq, pool_paths[cur ^ 1])
        pool_build_start = time.perf_counter()

        max_steps = args.batches_per_epoch or None
        for epoch in range(args.epochs):
            frozen = HDF5PatchDataset(pool_paths[cur], transform=to_tensor)
            # reshuffle the frozen pool every epoch; optionally cap the steps
            if max_steps is not None:
                sampler = torch.utils.data.RandomSampler(
                    frozen, replacement=True,
                    num_samples=max_steps * args.batch)
                loader = DataLoader(frozen, batch_size=args.batch,
                                    sampler=sampler,
                                    num_workers=args.num_workers)
            else:
                loader = DataLoader(frozen, batch_size=args.batch, shuffle=True,
                                    num_workers=args.num_workers)

            t = time.perf_counter()
            tr = train_one_epoch(loader, model, opt, device)
            va = validate(val_loader, model, device)
            dt = time.perf_counter() - t
            building = "building" if not future.done() else "READY"
            print(f"epoch {epoch:2d}  pool {'AB'[cur]}  "
                  f"train {tr:.4f}  val {va:.4f}  {dt:5.1f}s  "
                  f"[next pool: {building}]")

            # swap as soon as the next pool has landed; then start the following
            if future.done():
                future.result()                 # surface build errors
                built_s = time.perf_counter() - pool_build_start
                cur ^= 1                         # read the freshly built pool
                build_seq += 1
                future = pool_exec.submit(
                    build_pool, str(cube_path), variables, args.pool_size, 0,
                    build_seq, pool_paths[cur ^ 1])
                pool_build_start = time.perf_counter()
                print(f"  ↻ swapped to pool {'AB'[cur]} "
                      f"(built in ~{built_s:.1f}s in the background); "
                      f"building the next one now")

        # a queued build can be cancelled outright; one already running finishes
        # while the with-block's shutdown(wait=True) blocks on it — so surface
        # any error it hit instead of dropping it silently
        if not future.cancel():
            pending = future
        else:
            pending = None
    if pending is not None:
        pending.result()

    if owns_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"cleaned up {workdir}")
    else:
        print(f"left artifacts in {workdir}")


if __name__ == "__main__":
    main()
