# zarrpatch vs the ecosystem (2026-09-24)

Produced by `benchmarks/ecosystem_bench.py` (`pixi run -e ecosystem`); raw
numbers go to `results_ecosystem.json` (not committed), full tables are in
`results_ecosystem.table.md` and `results_ecosystem_cache128.table.md`.

## Setup

- **Cube:** `sst(time=200, lat=1200, lon=1200)` float32 (1.1 GiB decoded),
  smooth random field, zarr v3, default codec. Chunks `(10, 100, 100)`;
  *sharded* store packs them into `(50, 300, 300)` shards, *unsharded* store
  has one file per chunk. On a GPFS parallel filesystem.
- **Patches:** `5 × 64 × 64` (80 KiB). **random**: 20,000 uniform corners
  per epoch, a fresh draw each epoch, *identical* across loaders. **grid**:
  the full non-overlapping tiling (12,960 patches), shuffled each epoch.
- **Loading:** a torch `DataLoader`, batch 32, 0 or 8 workers
  (`forkserver`, `persistent_workers=True`, `OMP_NUM_THREADS=1`). Each
  configuration runs 3 epochs on one loader. Epoch 1 follows a best-effort
  page-cache eviction and includes worker start-up. **Warm** is the best of
  epochs 2–3.
- **CPU:** user + system seconds of the main process (all threads) plus every
  worker. **patches per CPU-second** is the efficiency figure. Wall-clock
  rates reward loaders that use more cores (TensorStore decodes on a pool of
  every core by default).
- **Host:** a shared 16-core compute node, whose load average varied between
  ~15 and ~60 during the run. Wall-clock numbers carry that noise. CPU-time
  numbers are much steadier.

## Headline (warm epochs)

patches/s and patches per CPU-second (in parentheses):

| loader | random, sharded, 0 w | random, sharded, 8 w | grid, unsharded, 8 w |
|---|---:|---:|---:|
| xarray `isel` per patch | 531 (262) | 2,061 (197) | 4,874 (486) |
| zarr-python per patch | 492 (251) | 1,882 (186) | 3,700 (437) |
| xbatcher | — | — | 4,381 (457) |
| TensorStore per patch | 2,500 (850) | 8,483 (686) | 11,621 (1,002) |
| TensorStore, batch of concurrent reads (all cores) | 8,490 (818) | 12,136 (795) | 15,622 (1,060) |
| TensorStore + 128 MiB cache (≈11% of cube) | 1,952 (666) | 6,749 (630) | 9,968 (897) |
| TensorStore + 1 GiB cache (≈ whole cube, per process) | 8,338 (4,651) | 31,390 (3,350) | 33,768 (3,661) |
| **zarrpatch**, default grouping (shard; chunk on the unsharded store) | 6,147 (3,273) | 19,572 (2,188) | 10,818 (1,147) |
| **zarrpatch `group_block`** (unsharded store) | — | — | 18,529 (2,064) |
| **zarrpatch, reads via TensorStore, 1 thread**¹ | 11,414 (5,495) | 31,338 (3,604) | 26,161 (3,007) |

¹ with `group_block` on the unsharded store.

## Findings

1. **Grouping multiplies CPU efficiency on the same decode engine.** On
   zarr-python, per-patch reads give 190–510 patches per CPU-second and
   zarrpatch gives 980–3,300 (3–13× in each configuration). On TensorStore,
   per-patch reads give 690–1,130 and zarrpatch's grouping gives 3,000–5,500
   (3–7×). This holds for random and grid sampling, sharded and unsharded
   stores, and 0 or 8 workers.
2. **The fastest per-patch loader, TensorStore issuing a batch of reads
   concurrently, wins on wall-clock only by spending cores.** Single-process,
   it reached ~8.5–10.6k patches/s using about 10 cores' worth of CPU (25 s
   of CPU for a 2.3 s epoch). Held to one decode thread, it drops to ~1.2–1.7k
   patches/s. zarrpatch on TensorStore with one thread beats that all-core
   throughput on random sampling and comes within ~12% of it on the grid,
   using a quarter to a sixth of the CPU.
3. **A chunk cache helps only when it holds the working set.** A 1 GiB
   TensorStore cache holds this whole cube in every worker. That makes it an
   in-memory dataset, and it is then as efficient as grouped reads. At 128 MiB
   (~11% of the cube) it falls back to per-patch cost (630–950 patches per
   CPU-second). Real training cubes are much larger than RAM, so that is the
   regime that matters. Grouping gets the same chunk reuse with memory bounded
   by one block group.
4. **Grouping by one chunk on an unsharded store underperforms, as
   expected.** zarrpatch's default chunk grouping on the unsharded store
   reached 980–1,450 patches per CPU-second. With
   `group_block=(20, 300, 300)` it reached 2,060–3,280 (≈1.8–2.5×), on par
   with the sharded store.
5. **zarr-python is now zarrpatch's bottleneck.** At equal grouping, reading
   through TensorStore instead of zarr-python is ~1.4–1.7× more efficient per
   CPU-second, and up to 1.9× faster on wall-clock. Grouped reads spend most of
   their time in zarr-python's per-chunk overhead, not in codecs.
6. **xbatcher and plain xarray are the least efficient (190–590 patches per
   CPU-second)**, roughly 4–12× below zarrpatch on the same store. xbatcher only enumerates
   regular tilings, so it could not run the random scenario at all.
7. **Where zarrpatch loses: start-of-epoch latency.** With 8 workers, a warm
   epoch's first batch arrives after 0.3–1.5 s for zarrpatch, versus ~0.05 s
   for map-style loaders. At these epoch sizes the 2048-patch shuffle buffer
   is close to a worker's whole share (2,500 random or 1,620 grid patches), so
   a worker yields almost nothing until it has read almost everything. The
   cost is a fixed stall per epoch and matters less on multi-million-patch
   epochs. It is also visible in epoch 1, where worker start-up plus buffer
   fill reached ~10 s on the grid scenario.

## Check: the shipped `TensorStoreReader`

The "reads via TensorStore" rows above came from a benchmark-only reader.
After `backend="tensorstore"` shipped, all zarrpatch variants were rerun in
one session, so both backends saw the same node load. Load average was ~170
then, versus ~15–60 for the tables above, so absolute rates are ~25% lower.
Full numbers are in `results_ecosystem_tsreader.table.md`. Warm patches per
CPU-second, zarr backend → TensorStore backend, at equal grouping (shards, or
`group_block` on the unsharded store):

| scenario | store | 0 workers | 8 workers |
|---|---|---:|---:|
| random | sharded | 2,401 → 3,984 (1.66×) | 1,797 → 2,981 (1.66×) |
| random | unsharded | 2,436 → 3,165 (1.30×) | 1,850 → 2,601 (1.41×) |
| grid | sharded | 2,335 → 3,493 (1.50×) | 1,567 → 2,541 (1.62×) |
| grid | unsharded | 2,418 → 3,169 (1.31×) | 1,805 → 2,592 (1.44×) |

The shipped reader keeps the 1.3–1.7× CPU-efficiency gain, and warm
throughput is 1.4–1.9× higher.

## What this does not show

- **A guaranteed cold GPFS read.** `posix_fadvise` cannot evict GPFS's
  pagepool without admin rights, so "epoch 1" is only best-effort cold. Its
  numbers are dominated by worker start-up and node noise. Treat them as
  indicative.
- **Remote object stores.** The whole comparison uses local files on GPFS.
- **WebDataset / pre-materialized shards.** Not
  compared here.
- **Other workloads.** These are one synthetic cube and one patch shape;
  sparse sampling, large patches and multi-variable samples will shift the
  ratios.
