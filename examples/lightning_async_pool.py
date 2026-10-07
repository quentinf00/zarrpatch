"""PyTorch Lightning recipe: async double-buffered materialize as a Callback.

Packages the mechanic from ``async_double_buffer.py`` for a Lightning training
loop. The work splits into two responsibilities, placed the idiomatic way:

* **``AsyncMaterializedPoolCallback``** owns the async *lifecycle* — it
  cold-builds the first pool, launches the background builder process, and tears
  everything down at ``fit`` end. Cross-cutting lifecycle is exactly what a
  ``Callback`` is for, and callbacks deliberately do NOT own dataloaders.
* **``MaterializedPoolDataModule.train_dataloader()``** owns *which pool this
  epoch*. With ``Trainer(reload_dataloaders_every_n_epochs=1)`` Lightning
  re-calls it at every epoch boundary, so it swaps to the freshly built pool and
  reshuffles the frozen patches for free. The swap is driven here (pull-based)
  rather than in a callback hook, so it can never race Lightning's dataloader
  reload.

The seam between the two is one shared ``PoolBuffer`` handle passed to both.

Why this is DDP-safe by construction
------------------------------------
* Each rank builds and reads its OWN pool files (``pool_{rank}_{A,B}.h5``), so
  there is no shared file to guard with a barrier, and ranks legitimately see
  different data (as a ``DistributedSampler`` would give).
* The swap is a deterministic function of the epoch number (``swap_every``), NOT
  of ``future.done()`` — so every rank swaps on the same epoch and the collective
  ops in the fit loop never desync.
* ``Trainer(limit_train_batches=...)`` fixes the step count per epoch, keeping
  all ranks in lockstep regardless of pool contents.
* The flip side of the deterministic swap: if a build takes longer than
  ``swap_every`` epochs, ``ensure_current`` blocks at the epoch boundary until
  it lands (unlike ``async_double_buffer.py``, which just keeps training).
  Size ``swap_every`` so one build comfortably fits inside it.

The alternative — one shared pool built on ``global_zero`` then a
``trainer.strategy.barrier()`` before any rank reads it — is noted inline;
per-rank pools are simpler and usually the better trade.

Run (needs ``lightning``, which is NOT a zarrpatch dependency)::

    pixi run -e lightning python examples/lightning_async_pool.py
    pixi run -e lightning python examples/lightning_async_pool.py \
        --pool-size 64000 --batches-per-epoch 1000 --epochs 40 --swap-every 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import lightning as L
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit(
        "this example needs PyTorch Lightning; install the optional env with "
        "`pixi run -e lightning python examples/lightning_async_pool.py` "
        "(or `pip install lightning`)."
    ) from exc

from zarrpatch import HDF5PatchDataset

# reuse the plain-PyTorch building blocks so the mechanic stays identical;
# importing from the sibling module gives the pool builder a clean qualified
# name for spawn-pickling into the background process.
from async_double_buffer import (
    PATCH_SIZE,
    VARIABLE,
    TinyAutoencoder,
    build_pool,
    build_synthetic_cube,
    n_possible_patches,
    to_tensor,
)


# --------------------------------------------------------------------------- #
# the shared handle: two ping-pong pool files + one background builder process
# --------------------------------------------------------------------------- #
class PoolBuffer:
    """Owns the two per-rank pool files and the background materialize process.

    Deterministic swap: :meth:`ensure_current` moves ``cur`` to the target index
    the epoch schedule asks for, submitting the next background build each time.
    Blocking on the pending build only ever waits for *this rank's* own build.
    """

    def __init__(
        self,
        cube_path: str,
        variables: List[str],
        pool_size: int,
        workdir: Path,
        rank: int = 0,
        seed: int = 0,
    ):
        self.cube_path = cube_path
        self.variables = variables
        self.pool_size = pool_size
        # fold the rank into the seed: every rank must draw a DIFFERENT corner
        # set, otherwise all ranks train on identical pools and the per-rank
        # files are pointless (a DistributedSampler-style split, not N copies)
        self.seed = seed + rank
        self.paths = [
            str(workdir / f"pool_{rank}_A.h5"),
            str(workdir / f"pool_{rank}_B.h5"),
        ]
        self.cur = 0
        self.build_seq = 0
        self._exec: Optional[ProcessPoolExecutor] = None
        self._future: Optional[Future] = None
        self._started = False

    def _submit_next(self) -> None:
        """Kick off the background build of the pool NOT currently read."""
        assert self._exec is not None
        self.build_seq += 1
        self._future = self._exec.submit(
            build_pool, self.cube_path, self.variables, self.pool_size,
            self.seed, self.build_seq, self.paths[self.cur ^ 1],
        )

    def start(self) -> float:
        """Cold-build the first pool synchronously, then launch the builder."""
        if self._started:
            return 0.0
        self._exec = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"))
        t = time.perf_counter()
        build_pool(self.cube_path, self.variables, self.pool_size, self.seed,
                   self.build_seq, self.paths[self.cur])
        cold_s = time.perf_counter() - t
        self._submit_next()  # start building the OTHER pool in the background
        self._started = True
        return cold_s

    def ensure_current(self, target: int) -> None:
        """Advance ``cur`` to ``target`` (0/1), swapping pools as needed."""
        if not self._started:
            self.start()
        while self.cur != target:
            if self._future is not None:
                self._future.result()  # block until the next pool is complete
            self.cur ^= 1
            self._submit_next()

    @property
    def current_path(self) -> str:
        return self.paths[self.cur]

    def shutdown(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._exec is not None:
            # cancel_futures drops queued builds, but a build already running
            # in the child cannot be cancelled — wait for it so nothing is
            # still writing pool files when the caller deletes the workdir
            self._exec.shutdown(wait=True, cancel_futures=True)
        self._exec = None
        self._started = False


# --------------------------------------------------------------------------- #
# LightningModule: a toy autoencoder over the patch tensor
# --------------------------------------------------------------------------- #
class LitAutoencoder(L.LightningModule):
    def __init__(self, channels: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyAutoencoder(channels)

    def _step(self, batch: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.model(batch), batch)

    def training_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _idx):
        self.log("val_loss", self._step(batch), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# --------------------------------------------------------------------------- #
# DataModule: owns "which pool this epoch" + the fixed validation snapshot
# --------------------------------------------------------------------------- #
class MaterializedPoolDataModule(L.LightningDataModule):
    def __init__(
        self,
        buffer: PoolBuffer,
        val_path: str,
        swap_every: int,
        batch: int,
        num_workers: int,
    ):
        super().__init__()
        self.buffer = buffer
        self.val_path = val_path
        self.swap_every = swap_every
        self.batch = batch
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        # deterministic, DDP-safe: the target pool is a function of the epoch,
        # so every rank agrees; pull-based here so it cannot race the reload.
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        target = (epoch // self.swap_every) % 2
        self.buffer.ensure_current(target)
        frozen = HDF5PatchDataset(self.buffer.current_path, transform=to_tensor)
        # NOTE: keep persistent_workers=False here — the ping-pong relies on
        # last epoch's workers being dead before the builder overwrites the
        # file they were reading
        return DataLoader(frozen, batch_size=self.batch, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        frozen = HDF5PatchDataset(self.val_path, transform=to_tensor)
        return DataLoader(frozen, batch_size=self.batch, shuffle=False,
                          num_workers=self.num_workers)


# --------------------------------------------------------------------------- #
# Callback: owns the async lifecycle (cold start + teardown)
# --------------------------------------------------------------------------- #
class AsyncMaterializedPoolCallback(L.Callback):
    def __init__(self, buffer: PoolBuffer):
        self.buffer = buffer

    def on_fit_start(self, trainer, pl_module) -> None:
        cold_s = self.buffer.start()
        if trainer.is_global_zero and cold_s:
            print(f"[callback] cold-built first pool in {cold_s:.1f}s; "
                  "background builder running")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            pool = "AB"[self.buffer.cur]
            print(f"[callback] epoch {trainer.current_epoch}: reading pool "
                  f"{pool} ({Path(self.buffer.current_path).name})")

    def on_fit_end(self, trainer, pl_module) -> None:
        self.buffer.shutdown()

    # teardown also fires on exceptions / KeyboardInterrupt
    def teardown(self, trainer, pl_module, stage: str) -> None:
        self.buffer.shutdown()


# --------------------------------------------------------------------------- #
def main() -> None:
    from zarrpatch import RandomSampler, ZarrPatchDataset, materialize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-size", type=int, default=4096)
    ap.add_argument("--batches-per-epoch", type=int, default=0,
                    help="0 = whole pool each epoch; realistic: 1000")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--swap-every", type=int, default=2,
                    help="epochs to train on a pool before swapping to the "
                         "freshly built one (deterministic, DDP-safe)")
    ap.add_argument("--val-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="zarrpatch_lightning_"))
    cube_path = workdir / "cube.zarr"
    val_path = str(workdir / "val.h5")
    sizes = {"time": 48, "lat": 384, "lon": 384}
    variables = [VARIABLE]

    build_synthetic_cube(cube_path, sizes)
    print(f"cube {dict(sizes)} → {n_possible_patches(sizes):,} possible "
          f"{PATCH_SIZE} patches; workdir {workdir}")

    # fixed validation snapshot, frozen once (uncompressed) and reused
    val_ds = ZarrPatchDataset(
        str(cube_path), PATCH_SIZE, variables=variables,
        sampler=RandomSampler(args.val_size),
        group_by_block=True, shuffle=False, prefetch=1, seed=12345,
    )
    materialize(val_ds, val_path, compression=None, overwrite=True)

    # one PoolBuffer, shared by the callback (lifecycle) and datamodule (reads).
    # rank=0 here; under DDP use trainer.global_rank so each rank owns its files.
    buffer = PoolBuffer(str(cube_path), variables, args.pool_size, workdir,
                        rank=0)
    dm = MaterializedPoolDataModule(
        buffer, val_path, args.swap_every, args.batch, args.num_workers)
    model = LitAutoencoder(PATCH_SIZE["time"])

    trainer = L.Trainer(
        max_epochs=args.epochs,
        reload_dataloaders_every_n_epochs=1,       # re-call train_dataloader()
        limit_train_batches=args.batches_per_epoch or 1.0,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[AsyncMaterializedPoolCallback(buffer)],
    )
    try:
        trainer.fit(model, datamodule=dm)
    finally:
        buffer.shutdown()
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"cleaned up {workdir}")


if __name__ == "__main__":
    main()
