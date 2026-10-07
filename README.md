# zarrpatch

Fast random-patch dataloading from geophysical datacubes (zarr v2/v3, netCDF,
kerchunk/virtualizarr references) for neural-network training on HPC.

**The problem.** Training on random patches of a chunked datacube naively reads
one patch at a time; each read fetches and decompresses at least one whole
chunk, so most decompressed bytes are thrown away and, on parallel filesystems
(Lustre/GPFS), each patch costs several small-file metadata operations.

**What zarrpatch does.**

1. Draws all patch corners for the epoch up front (random, strided grid, or an
   explicit coordinate list).
2. Groups corners by storage block — the **zarr v3 shard** if present, else the
   chunk — and visits blocks in random order, reading each block region **once**
   and slicing all of its patches out of memory.
3. Pipes the patch stream through a **shuffle buffer** so samples from the same
   block don't end up in the same batch.

On the bundled benchmark (137 MiB cube, 2000 random 5×64×64 patches) this is
**~12x faster than naive netCDF loading and ~8x faster than naive zarr**, at
identical sample content. See `benchmarks/results.md`.

### Against the alternatives

`benchmarks/results_ecosystem.md` compares zarrpatch on the same corners and
store (1.1 GiB zarr v3 cube on GPFS, 5×64×64 patches, 0 or 8 persistent
workers) with plain xarray `isel`, zarr-python per-patch reads, xbatcher, and
TensorStore. The TensorStore variants are per-patch, concurrent batches, and
with a chunk cache. CPU time is counted, not just wall-clock. Warm epochs,
patches per CPU-second:

- xarray / zarr-python / xbatcher per patch: **190–590**
- TensorStore per patch (with or without concurrency, or with a cache ≈11% of
  the cube): **630–1,130**
- zarrpatch: **980–3,300**, or 1,930–3,300 when grouping by several chunks
  (shards or `group_block`)
- zarrpatch with `backend="tensorstore"`: **3,000–5,500**

On the same engine, grouping is worth 3–13×. A TensorStore chunk cache gets
there only when it can hold the whole cube in every worker. With the default
zarr-python backend, per-chunk Python overhead is the main cost, which is
why the TensorStore backend exists (below). zarrpatch's weak
spot is a start-of-epoch stall while the shuffle buffer fills (0.3–1.5 s per
warm epoch here). Rerun with `pixi run -e ecosystem ecosystem-bench`.

New to chunked patch loading? Start with
[`docs/loading-primer.md`](docs/loading-primer.md): where loading time goes, and what
formats and strategies such as chunking, sharding, memmaps and shuffle buffers address.

## In scope / out of scope

Everything zarrpatch does follows from one contract: **a variable is a regular
N-D array on named dims, stored in uniform rectangular chunks/shards; a patch is
a fixed-length rectangular window; corners are grouped by
`index // block_size`.** That contract is what makes read-each-block-once, the
DDP/worker partition, the shuffle buffer, and weighted reconstruction possible.
The dividing line for any new use case is simply how well it fits that contract.

**In scope — works today (possibly with a small helper):**

- **Regular N-D fields** — the core case (`sst(time,lat,lon)`, `temp(time,depth,
  lat,lon)`, …), any number of patched dims.
- **Mixed-dimension multi-variable samples** — static (`bathymetry(lat,lon)`),
  dynamic, 2-D and 3-D variables combined into one patch dict; a variable is
  read only over the patch dims it actually has, and dims not in `patch_size`
  are read in full extent. See *Mixed-dimension variables* below.
- **1-D DGGS / HEALPix grids** — data on a `cell` dim (`var(time, cell)`).
  Because dims are just names, sampling fixed-length `cell` windows is fully
  supported. In **nested** ordering a `4ᵏ`-aligned `cell` range is a
  spatially-contiguous superpixel, so patches are spatially local. See *HEALPix
  / DGGS grids* below.
- **Contiguous-ragged swaths, fixed-length along-track windows** — swaths
  concatenated into one long `obs` dim, sampled as fixed-length windows
  (`patch_size={"obs": 256}`). Keep windows inside one acquisition with a
  boundary-aware `ListSampler` or a `MaskedSampler`.
- **Single hierarchical group** — point the path at the subgroup
  (`store.zarr/subgroup`); its top-level arrays are read as usual.

**Out of scope — needs a different layer, not a minor change:**

- **Variable-length / per-acquisition outputs** — ragged swaths loaded whole
  (each a different length). Violates the fixed `patch_size` contract and torch
  collation; `MultiCubeReader` requires uniform unit shapes by design.
- **Selection over irregular physical coordinates** — "give me swath samples in
  this lat/lon/time box" needs a spatial/temporal index (KD-tree, R-tree,
  binning). zarrpatch groups purely in storage-index space and has no
  coordinate index.
- **Arbitrary spherical neighbourhoods on HEALPix** — a general k×k window
  around a point is not a contiguous 1-D `cell` range; only nested-aligned
  superpixels are.
- **Heterogeneous grids in one sample** — combining variables across groups
  with different grids/resolutions (e.g. multi-resolution DataTree). Dims must
  align across the variables of one dataset.

For the out-of-scope cases, the right pattern is usually a separate
indexing/regularization layer that *feeds* zarrpatch once the data is on a
regular grid — not an extension of zarrpatch itself.

## Install / develop

```bash
pixi install
pixi run test    # pytest suite (default env includes the optional backends,
                 # TensorStore included)
pixi run bench   # generate synthetic stores + run the format comparison
pixi run -e ecosystem ecosystem-bench   # vs xarray / xbatcher / TensorStore
```

With pip, the core needs only `numpy` and `zarr`; optional extras are
`torch` (the dataset, planning and materialization), `tensorstore` (the faster
zarr backend), `netcdf`, `kerchunk`, `hdf5` and `lightning`.

## Usage

```python
from torch.utils.data import DataLoader
from zarrpatch import ZarrPatchDataset, RandomSampler, GridSampler, ListSampler

ds = ZarrPatchDataset(
    "cube.zarr",                          # zarr v2/v3 store, .nc file, or
                                          # kerchunk reference .json
    patch_size={"time": 5, "lat": 64, "lon": 64},
    variables=["sst", "ssh"],             # default: all vars with the patch dims
    sampler=RandomSampler(50_000),        # patches per epoch
    shuffle_buffer=2048,                  # patches held per worker for mixing
    seed=0,
)
loader = DataLoader(ds, batch_size=32, num_workers=4, persistent_workers=True)

for epoch in range(n_epochs):
    ds.set_epoch(epoch)                   # fresh corners + shuffling each epoch
    for batch in loader:                  # dict of tensors, e.g. batch["sst"]
        ...
```

Samplers:

- `RandomSampler(n)` — n uniform random corners, redrawn every epoch.
- `GridSampler(stride={"lat": 32, ...})` — all patches on a strided grid
  (default stride = patch size, i.e. a full non-overlapping tiling; the last
  patch per dim is clamped to the boundary).
- `ListSampler(corners)` — explicit index-space corners;
  `coords_to_indices(...)` converts physical coordinate points (e.g. lat/lon
  values) into centered, clamped corners.
- `MaskedSampler(base, mask, dims, max_invalid_frac=0.0)` — wraps another
  sampler and keeps only corners whose patch footprint is valid under a mask
  (e.g. an ocean mask, or `np.isfinite(field).mean("time") > 0.9`); random
  draws are topped up so the epoch size is preserved.
- `WeightedSampler(n, weight, dims)` — draws `n` corners **with replacement**,
  corner probability proportional to the mean of `weight` over the patch
  footprint on `dims` (patch-integrated via the same summed-area table as
  `MaskedSampler`, so a patch inherits weight from all its rows). `weight` is
  an array over a subset of the patch dims, exactly like the mask — e.g.
  latitude oversampling `weight = 1 + (f-1)*(|lat|/90)**2` over a 1-D `lat`, or
  per-store cluster resampling over a virtual `store` dim. Composes with
  `MaskedSampler(WeightedSampler(...), mask, mdims)`. Keep the weighted `dims`
  to 1–2 (corners are enumerated jointly over them). Caveat: because corners
  are drawn with replacement, the dataset's "every patch exactly once per
  epoch" guarantee becomes "every drawn corner exactly once" — duplicate
  corners are allowed and cheap (same block group, sliced once each).

Dims of a variable not in `patch_size` are read in full extent (e.g. a static
`mask(lat, lon)` alongside `sst(time, lat, lon)` patched over lat/lon).

Mixed-dimension variables — static + dynamic + 2-D + 3-D in one sample:

```python
# one dataset, variables of different rank; each is read only over the patch
# dims it has, and dims absent from patch_size are read in full extent.
ds = ZarrPatchDataset(
    "cube.zarr",
    patch_size={"time": 5, "lat": 64, "lon": 64},   # depth NOT patched
    variables=[
        "temp",        # (time, depth, lat, lon) -> patch (5, depth, 64, 64)
        "sst",         # (time, lat, lon)        -> patch (5, 64, 64)
        "bathymetry",  # (lat, lon)              -> patch (64, 64), no time axis
    ],
    sampler=RandomSampler(50_000),
)
for batch in DataLoader(ds, batch_size=32):
    batch["temp"]        # (32, 5, depth, 64, 64)
    batch["sst"]         # (32, 5, 64, 64)
    batch["bathymetry"]  # (32, 64, 64)
```

Each variable is read once per block group over its own bounding region, so the
static `bathymetry(lat, lon)` is not re-read per patch, and `temp`'s full depth
axis comes along for free (add `depth` to `patch_size` to crop it instead).
Torch's default collation stacks each key independently, so the differing ranks
are fine. Two things to keep aligned: pass `variables` explicitly (the default
picks only variables that contain *every* patch dim, which would drop a static
`bathymetry`), and **list the most finely-chunked variable first** — block
grouping keys off `variables[0]`'s chunk layout along the patched dims, and
warns if another variable's chunks differ.

HEALPix / DGGS grids — 1-D `cell` dimension:

```python
import healpy as hp
from zarrpatch import ZarrPatchDataset, RandomSampler, ListSampler

# data stored as var(time, cell), cell length 12*nside**2 in NESTED ordering,
# chunked along cell. A 4**k-aligned cell window is a spatially-contiguous
# superpixel (a coarse pixel's descendants), so patches are spatially local.
ds = ZarrPatchDataset(
    "healpix_cube.zarr",
    patch_size={"time": 5, "cell": 4 ** 4},   # 256-cell nested superpixels
    variables=["sst"],
    sampler=RandomSampler(50_000),
)

# to sample around given lat/lon points, resolve them to nested cell indices
# yourself (the cell index is not monotonic in lat/lon, so coords_to_indices
# does not apply) and snap to the 4**k boundary, then use ListSampler:
nside, k = 1024, 4
block = 4 ** k
cells = hp.ang2pix(nside, lons, lats, nest=True, lonlat=True)
corners = [{"time": 0, "cell": int(c) // block * block} for c in cells]
ds = ZarrPatchDataset(
    "healpix_cube.zarr", patch_size={"time": 5, "cell": block},
    variables=["sst"], sampler=ListSampler(corners),
)
```

Only nested-aligned superpixels are spatially contiguous — a general spherical
neighbourhood around a point is not a contiguous `cell` range, so window-around-
a-point sampling is out of scope (see *In scope / out of scope*).

Train/val/test splits — one store, no per-split copies:

```python
train = ZarrPatchDataset(path, patch, dim_slices={"time": (0, 800)}, ...)
val   = ZarrPatchDataset(path, patch, dim_slices={"time": (800, 900)}, ...)
```

`dim_slices` restricts the dataset to an index range per dim (`slice(a, b)` or
`(a, b)` tuples); everything — sampler corners, `ListSampler` input, the
`"corner"` output — then lives in the sliced index space, as if the cube had
been cut with `isel`. Reads stay aligned to the underlying storage blocks
even when the slice start isn't. Sliced dims not in `patch_size` are read
over the restricted range only.

Zipped stores and many stores as one cube:

```python
from zarrpatch import ZarrPatchDataset, MultiCubeReader, expand_units, WeightedSampler

# a single .zarr.zip is opened transparently through zarr's ZipStore
ds = ZarrPatchDataset("storm.zarr.zip", patch_size={"step": 16, "y": 64, "x": 64})

# 932 per-storm zips with ragged init_time, as one virtual cube: expand each
# store into one unit per forecast cycle, then concatenate along a new "store"
# dim (block size 1 along it, so reads never cross stores)
reader = MultiCubeReader(expand_units(paths, "init_time"), dim="store")
ds = ZarrPatchDataset(
    reader,                                   # a CubeReader, not a path
    patch_size={"store": 1, "step": 16, "y": 64, "x": 64},  # store patch must be 1
    variables=["u10", "v10"],
    sampler=WeightedSampler(50_000, cluster_weight, ["store"]),  # cluster resampling
)
```

`ZarrPatchDataset`'s first argument accepts either a path (opened via
`open_cube`) or an already-built `CubeReader`. `MultiCubeReader(sources, dim)`
concatenates same-schema cube *units* along a new leading dim (default
`"store"`); a source is a path or a `(path, {dim: index})` pair whose selected
indices are squeezed out — use `expand_units(paths, dim)` to turn stores with a
ragged/reducible dim (e.g. different `init_time` lengths) into uniform units.
Stores are opened lazily behind an LRU (`cache_size=8`); unit 0 fixes the
reference schema and other units are validated on first read (mismatched
variable/dims/shape/dtype error, mismatched chunking warns once). `patch_size`
must set the virtual dim to `1`, so each block group — and each patch — reads
exactly one unit; `dim_slices={"store": (a, b)}` restricts to a unit range
(e.g. a storm-level train/val split) with no copies.

TensorStore backend — same patches, less CPU:

```python
ds = ZarrPatchDataset("cube.zarr", patch_size, backend="tensorstore")
# or, to tune decode threads (default 1 per process, i.e. per worker):
from zarrpatch import TensorStoreReader
ds = ZarrPatchDataset(TensorStoreReader("cube.zarr", threads=4), patch_size)
loader = DataLoader(ds, batch_size=32, num_workers=8,
                    multiprocessing_context="forkserver")   # required, see below
```

`pip install zarrpatch[tensorstore]`. Metadata (variables, dims, shapes,
shard/chunk layout) still comes from zarr-python, so both backends describe a
store identically and yield identical patches. Only region reads go through
TensorStore's C++ engine, with all variables of a group read concurrently. It
covers zarr v2/v3, sharded or not, local directories and `.zip` stores
(`s3://`, `gs://` and `http(s)://` URLs are passed through, untested). It does
not cover netCDF or kerchunk references. Each array is cross-checked against
zarr-python on first read, and anything TensorStore cannot open or reads
differently raises `ValueError` pointing back to `backend="zarr"`. It stays
opt-in; the zarr backend remains the default.

**Fork:** TensorStore aborts any process forked after it has started its
threads. The reader therefore opens nothing at construction and opens handles
lazily in each worker. But if you read in the main process (e.g.
`num_workers=0`, or `materialize`) and *then* start a DataLoader with the
default `fork` method, its workers abort. Use
`multiprocessing_context="forkserver"` or `"spawn"`. The reader warns once
when it reads in the main process.

Inference — exhaustive scan + weighted reconstruction:

```python
from zarrpatch import Reconstructor

ds = ZarrPatchDataset(
    path, patch_size, variables=["sst"],
    sampler=GridSampler(stride={"lat": 32, "lon": 32}),  # overlapping scan
    shuffle=False, return_corners=True,
    dim_slices={"time": (800, 900)},
)
rec = Reconstructor.from_dataset(ds)          # weight="hann" by default
for batch in DataLoader(ds, batch_size=32, num_workers=4):
    rec.add(batch["corner"], model(batch["sst"]))
cube = rec.result()                           # NaN where nothing contributed
```

Overlapping predictions are averaged with a per-position weight window
(`"hann"` default, also `"triangle"`, `"uniform"`, or a custom array), so
patch centers dominate and seams blend smoothly. Predictions may have extra
leading dims the input doesn't (channels, ensemble members, ...) — only the
scanned (patch) dims are reconstructed; if the model changes the scanned
extent (e.g. center crop), build `Reconstructor(dim_sizes, out_patch_size)`
with the output extent and shift the corners accordingly. Under DDP, `merge()`
the per-rank reconstructors (or all-reduce `rec.sum` / `rec.weight_sum`)
before `result()`.

Distribution & pipelining:

- The epoch's shuffled block groups are cut into exact-size contiguous pieces,
  first across `torch.distributed` ranks, then across each rank's dataloader
  workers (a group straddling a cut is read by both sides). Every rank yields
  **exactly the same number of patches**, and the same per-worker counts, so
  every rank runs the same number of batches — a rank that finishes early
  would otherwise deadlock the others' gradient all-reduce at epoch end. The
  `N mod world_size` remainder is dropped (`ddp_remainder="drop"`, default) or
  padded by re-yielding the first patches (`"pad"`). Each patch is yielded at
  most once per epoch across the job (exactly once without DDP or with an
  epoch divisible by `world_size`).
- `len(ds)` is the per-rank patch count under DDP (like `DistributedSampler`,
  which is what Lightning and LR schedulers expect) and the whole epoch
  otherwise; `ds.epoch_size()` is always the global count.
- `set_epoch` stores the epoch in shared memory, so it also works with
  `persistent_workers=True`.
- Each worker reads the next block group in a background thread while the
  current one is consumed (`prefetch=1`; set 0 to disable).
- Blocks whose bounding-region read would exceed `max_block_bytes` (default
  1 GiB — e.g. contiguous netCDF, where the "block" is the whole array) are
  split into smaller virtual blocks, with a warning.
- The shuffle buffer holds `shuffle_buffer` decoded patches in memory per
  worker (e.g. 2048 × 80 KiB patches ≈ 160 MiB). It also **bounds batch
  mixing**: an `IterableDataset` batch is collated from a single worker's
  stream, so it can only mix patches that passed through that worker's buffer.
  `"auto"` = `max(2048, 4 × largest patches-per-block)`; `"io"` = only the
  `4 × patches-per-block` minimum that keeps one decoded block from landing
  whole in a batch — an I/O rule, not a mixing guarantee. Check what you get with
  `ds.plan(batch_size=...)`.

Multiprocessing safety (avoid the fork/OpenMP hang):

```python
from zarrpatch import ZarrPatchDataset, worker_init_fn

loader = DataLoader(
    ds, batch_size=32, num_workers=4,
    multiprocessing_context="forkserver",   # don't inherit a locked libomp mutex
    worker_init_fn=worker_init_fn,          # pin torch/blosc to 1 thread/worker
)
```

A `DataLoader` with the default `fork` start method can hang forever if a worker
is forked while a native thread pool's lock (libomp/BLAS/blosc) is held in the
parent — likely here because decode runs through numcodecs/numpy and
`MaskedSampler`/`WeightedSampler` do numpy reductions in the main process right
before the loader forks. Set `OMP_NUM_THREADS=1` **in your launcher** (before
numpy imports) and use `multiprocessing_context="forkserver"`. `worker_init_fn`
handles CPU oversubscription but runs too late to prevent the hang. Full
analysis and fixes: see the `worker_init_fn` docstring.

With `backend="tensorstore"`, `forkserver` (or `spawn`) is **required** once
the main process has read anything: TensorStore aborts forked children after it
has started its threads (see *TensorStore backend*). Passing `timeout=...` to
the `DataLoader` turns a stuck worker into an exception instead of a silent
stall.

Inspect the exact epoch plan before training:

```python
from zarrpatch import ZarrPatchDataset, analyze_layout

plan = ds.plan(world_size=4, num_workers=8, batch_size=32)
print(plan.summary())
plan.to_json("plan.json")

# Convenience form for a store that is not already open:
report = analyze_layout(
    "cube.zarr", {"time": 5, "lat": 64, "lon": 64},
    world_size=4, num_workers=8,
)
```

Planning samples corners and inspects metadata but never reads array payloads.
It reports patch/group distributions, per-shard counts, logical, read
(bounding-region) and **decoded** (chunk-aligned) bytes, decode amplification,
in-flight memory, variable layout mismatches, estimated store opens, and
actionable warnings. With `batch_size`, it replays every worker's exact shuffle
stream on block labels and reports distinct block groups per batch next to an
ideal global shuffle of the same epoch, warning when batches are dominated by
single blocks. The topology is explicit, so analysis does not depend on
whether it runs inside a DataLoader.

Materialize — freeze a fixed corner set to HDF5:

```python
from torch.utils.data import DataLoader
from zarrpatch import ZarrPatchDataset, GridSampler, materialize, HDF5PatchDataset

val = ZarrPatchDataset(
    "cube.zarr", patch_size={"time": 5, "lat": 64, "lon": 64},
    sampler=GridSampler(stride={"lat": 64, "lon": 64}), shuffle=False,
)
materialize(val, "val.h5")                     # one sequential pass, single writer
frozen = HDF5PatchDataset("val.h5")            # map-style, opens lazily per worker
loader = DataLoader(frozen, batch_size=32, num_workers=4)
```

Snapshots are built and validated in a temporary sibling before atomic
publication. `HDF5PatchDataset` eagerly rejects incomplete, corrupt, or
unsupported snapshots; `validate_materialized("val.h5")` exposes the same
check directly. If the live dataset has a Python transform, pass a stable
caller-owned identity such as `materialize(ds, path, transform_id="assemble-v3")`;
the callable itself cannot be fingerprinted reliably.

When the corner set is fixed for a run — validation/test, or a once-drawn
weighted training pool — `materialize()` does one pass over a configured dataset
and writes an HDF5 snapshot (one dataset per variable, one sample per chunk, plus
a `corner` dataset). `HDF5PatchDataset` reads it back as a map-style
`torch.utils.data.Dataset` that composes with stock torch samplers
(`DistributedSampler` for DDP, `WeightedRandomSampler` over indices for a
reweighted frozen pool). The formats benchmark measured **~32× over live reads
through this exact API** (~1600 vs ~50 samples/s, frozen assembled sample; the
hand-packed single-tensor ceiling is ~73×) — a strictly better `.pt`-directory
replacement.

**Under DDP,** `materialize()` always writes the whole epoch, even inside an
initialized job where the live dataset yields only one rank's share. Call it
on one rank, barrier, then read it on every rank:

```python
if trainer.global_rank == 0:          # or dist.get_rank() == 0
    materialize(val, "val.h5", overwrite=True)
trainer.strategy.barrier()            # or dist.barrier()
frozen = HDF5PatchDataset("val.h5")   # + DistributedSampler on every rank
```
`compression=` is `None` (default, fastest), `"lzf"` (fast middle), or `"gzip"`
(= gzip-4, disk-bound). Decompression is the read bottleneck, so it trades speed
for size: uncompressed reads are by far the fastest, `"lzf"` gives a much
smaller file at a fraction of that speed, and `"gzip"` the smallest file at
the slowest read. **Default to `None`** for a
validation set you rerun often (speed dominates, the snapshot is transient);
reach for `"lzf"` only when disk is tight, and `"gzip"` only when the file must
be tiny (archival/transfer) and its read speed still clears your input pipeline.
`max_bytes=` is a pre-flight size refusal. Needs the optional extra
`pip install zarrpatch[hdf5]`. Read live instead when corners must be fresh every
epoch or the sampling/geometry still varies — that is what the live dataset is
for.

`materialize()` logs the storage impact before writing (an `INFO` record on the
`"zarrpatch"` logger: snapshot size, per-sample size, and the source read-region
size) and warns when the snapshot would be **larger than the region it is drawn
from** — the signal that patches overlap or the sampler draws with replacement,
so pixels are stored several times. When that fires, freeze a coarser,
*non-overlapping* granularity (drop the overlapping dim from `patch_size` and
slice it at read time in the `transform`) rather than materializing the
overlapping windows: e.g. for 16-step windows over a 43-step sequence, freeze
the whole sequence (`patch_size` without `step`) and slice the window on read,
storing each frame once instead of ~9×.

## Chunking & sharding guidelines

`recommend_layout()` encodes these rules and prints concrete shapes:

```python
from zarrpatch import recommend_layout
print(recommend_layout(
    dim_sizes={"time": 2000, "lat": 2400, "lon": 3600},
    patch_size={"lat": 128, "lon": 128},
    sampling="random",          # or "grid" (+ stride=...)
).summary())
```

The rules, and why:

1. **Chunk edge ≈ patch edge along patched dims.** Much smaller chunks multiply
   per-chunk request/decompress overhead; much larger chunks decompress data
   you throw away and force a bigger shuffle buffer (more patches per block).
   The chunk is the *decode* unit; reads must be *grouped* by several chunks
   (rule 4 or 5).
2. **Grid/stride sampling: align chunk edges to the stride** so patch reads
   fall on chunk boundaries and no chunk is decoded twice.
3. **Un-patched dims** are read in full for every patch — chunk them so the
   total chunk lands in the ~1–16 MiB sweet spot.
4. **Use zarr v3 shards on HPC.** Keep the chunk small (read granularity) and
   pack many chunks per shard file (~64–512 MiB) to keep the file count low:
   millions of tiny chunk files crush Lustre/GPFS metadata servers.
5. **Sharding only pays off with block-grouped reads.** A naive per-patch
   reader on a sharded store is *slower* (shard index indirection per tiny
   read, and whole-shard requests on some stores) — the benchmark shows 0.6x.
   Group reads by shard (zarrpatch's default) and the same store is >12x.
   **Without shards, pass `group_block`.** A group's read region extends
   `patch - 1` past its block, so grouping by one patch-sized chunk straddles
   the next chunk on every patched dim and decodes chunks up to `2^d` times.
   `ZarrPatchDataset(group_block={"lat": 512, ...})` groups by a virtual block
   of several chunks instead (≥ ~4 × (patch − 1) keeps the straddle under
   25%); `recommend_layout(use_shards=False)` prints a value, and the dataset
   warns when the default grouping would straddle heavily.
6. **Shuffle buffer size:** several times the patches-per-block count
   (≈ `n_patches × shard_volume / cube_volume` for random sampling). The
   benchmark shows its throughput cost is ~1–2%.
7. **Compression:** prefer fast codecs (zstd level ≤ 3, or lz4 + shuffle) —
   with grouped reads the pipeline is usually decompress-bound, not I/O-bound.

## Benchmarking utilities

```python
from zarrpatch import benchmark, print_results, results_to_markdown

r1 = benchmark(lambda: DataLoader(ds, batch_size=32), name="mine", repeats=2)
print_results([r1])           # rich table: wall, first-item latency, items/s, MiB/s
```

`benchmarks/run_bench.py` (`pixi run bench`) writes the same synthetic cube as
netCDF, zarr v2, zarr v3 and sharded zarr v3, then compares naive/grouped/
buffered loading with identical patch corners. Stores go to `benchmarks/data`
unless `ZARRPATCH_BENCH_DATA` is set: copy `.env.example` to `.env` (gitignored),
which `pixi` sources on activation. Results from this machine:

| loader | patches/s | speedup |
|---|---:|---:|
| netcdf / naive | 639 | 1.00x |
| zarr v2 / naive | 1132 | 1.77x |
| zarr v3 / naive | 959 | 1.50x |
| zarr v3 sharded / naive | 383 | 0.60x |
| zarr v3 / grouped, no buffer | 3356 | 5.25x |
| zarr v3 sharded / grouped, no buffer | 7456 | 11.67x |
| zarr v3 sharded / grouped + shuffle buffer | 7672 | 12.01x |
| zarr v3 sharded / grouped + buffer + prefetch | 8041 | 12.58x |

(medium cube — dims `{time: 100, lat: 600, lon: 600}` float32, chunks
`(10, 100, 100)`, shards `(50, 300, 300)`, 2000 random `5×64×64` patches,
single process, warm page cache. Rerun with `pixi run bench`; use
`--size large --workers 4 --fresh` for a more I/O-bound setting.)

### Startup and I/O profiling

`JsonlProfiler` is an opt-in trace for diagnosing slow DataLoader startup and
multi-store access. It writes one JSONL file per process, so DataLoader workers
need no logging queue or shared lock:

```python
from zarrpatch import JsonlProfiler, ZarrPatchDataset

profiler = JsonlProfiler("traces/run-001")  # detailed=True adds every read
ds = ZarrPatchDataset(
    "cube.zarr",
    patch_size={"time": 5, "lat": 64, "lon": 64},
    profiler=profiler,
)
loader = DataLoader(
    ds, batch_size=32, num_workers=8, persistent_workers=True,
    multiprocessing_context="forkserver",
)

# These main-process markers expose time spent before workers enter ds.__iter__.
profiler.event("loader_iter_requested")
it = iter(loader)
batch = next(it)
profiler.event("first_batch_received")
```

For a `MultiCubeReader`, pass the same profiler when constructing the reader
to capture its initial reference-schema open as well:

```python
reader = MultiCubeReader(sources, profiler=profiler)
ds = ZarrPatchDataset(reader, patch_size, profiler=profiler)
```

Summarize all worker files with:

```bash
python -m zarrpatch.profiling traces/run-001
python -m zarrpatch.profiling traces/run-001 --json
```

The default trace records iteration/planning, first-patch latency, store opens,
multi-store reader cache misses/hits/evictions, process and worker IDs, and
iteration completion. `detailed=True` additionally records every block-group
and variable read with elapsed time and decoded byte count; use it for short
diagnostic runs because per-read JSONL writes perturb a high-throughput epoch.

Times around `open_group`, metadata access, and array reads are filesystem
latencies as observed by zarrpatch. The trace cannot by itself attribute that
latency to GPFS metadata contention, page-cache misses, network delay, or
storage throttling; correlate it with filesystem/system telemetry when that
distinction matters.

## Troubleshooting

These are the bottlenecks we hit in practice, while training on real
multi-terabyte cubes and on multi-store datasets with hundreds of stores.

### Where the time goes

[`docs/loading-primer.md`](docs/loading-primer.md) walks through every stage a sample
passes through (metadata, I/O, decompression, slicing, shuffle buffer, collate,
worker transfer). It also covers the storage formats and loading strategies that
each address one of them, and ends with a measured per-stage breakdown
(`benchmarks/time_breakdown.py`). For your own runs, use the
[`JsonlProfiler`](#startup-and-io-profiling) and `ds.plan()`.

### Host memory grows with `num_workers`, or the job is OOM-killed

Peak host memory is roughly

    num_workers × (shuffle_buffer × transformed sample size
                   + whatever the dataset or transform loads per worker)
    + decoded block groups in flight

and every term has bitten us:

- **The shuffle buffer holds transformed samples.** If `transform` turns a
  patch into a multi-megabyte stack of channels, the default 2,048-sample buffer
  can grow to tens of GB *per worker* (we measured 30–37 GB RSS per worker). If
  the buffer is as large as a worker's whole share (`len(ds) / num_workers`),
  it is also slow. The worker reads its entire share before the first batch,
  so nothing overlaps with GPU compute. Dropping the buffer from 2,048 to 256
  cut time-to-first-batch from 66 s to 12 s. Keep `shuffle_buffer` well below
  the per-worker share, and check `ds.plan(batch_size=...)` that mixing is
  still acceptable.
- **Each worker gets its own copy of anything the dataset loads eagerly.** A
  climatology or normalization field of several GB, loaded in `__init__` or
  in the transform, is duplicated `num_workers` times. Open such data lazily
  inside each worker, or memory-map it read-only so workers share the page cache.
- **Worker generations overlap.** Training workers stay resident during
  validation, so peak memory is training workers plus validation workers. With
  `reload_dataloaders_every_n_epochs=1`, check that old workers are reaped.
- **Read amplification.** A group reads the bounding box of its patches. With
  sparse sampling on large shards, this decoded 3.6× (train) to 8.5× (val)
  more data than the patches needed. Check `ds.plan()` for decoded bytes, and
  use smaller shards or a `group_block` that matches the sampling density.
- **NUMA starvation looks like a slow loader.** On multi-socket nodes, the
  default allocation keeps memory on the job's local NUMA node. Two
  memory-heavy jobs on the same node exhausted it while the other nodes still
  had hundreds of GB free. Every allocation then went through failing
  compaction and reclaim, workers ran at ~97% *system* time inside the buffer
  copy, and a validation pass took 15 min to 2 h instead of seconds. The GPU
  was busy for under a quarter of the wall time. Symptoms are high `%sys`
  with no I/O wait and an exhausted node in `numastat` or
  `/proc/buddyinfo`. Fix: launch with `numactl --interleave=all`, and reduce
  memory using the points above.

### The worker transfer cap

Every sample a worker produces is collated and sent through shared memory to
the main process. A control dataset that returns in-memory arrays (no I/O at
all) shows the ceiling any reader can reach behind a worker `DataLoader`,
roughly **2–4 GiB/s**:

| control dataset, samples/s | in-process | 4 workers | 8 workers |
|---|---:|---:|---:|
| 0.23 MiB samples | 31k | 8.6k | 8.3k |
| 5 MiB samples | 337 | 535 | 844 |

Adding workers barely moves the small-sample ceiling, and in-process is faster
than workers for small samples. Live `ZarrPatchDataset` reads (hundreds to a
few thousand patches/s per worker) are far from this cap. It does bind for
materialized snapshots (`HDF5PatchDataset`) and other near-memory-speed
sources. With 8 workers, h5py already reaches ~75% of the small-sample cap and
essentially all of the large one, so a faster file format gains ≤1.3×. A large
win needs in-process reading with whole-batch reads (`num_workers=0`), which
measured ~9× faster on small samples. In that regime, the loader topology
matters more than the file format.

### First batch of every epoch is slow

- **Shuffle buffer fill.** A worker yields nothing until its buffer is full.
  On short epochs, where the buffer is close to the worker's share, the first
  warm batch arrives after 0.3–1.5 s, against ~0.05 s for map-style loaders.
  In the first epoch it is ~10 s including worker start-up. This is a fixed
  GPU-idle gap per epoch. A smaller buffer or longer epochs shrink it.
- **Many stores.** Workers open stores lazily, so on a dataset of hundreds of
  stores the cold first epoch is dominated by opens and cold cache. On a tiny
  development subset, worker start-up dominates: time to first batch grew from
  ~9 s with 0 workers to ~80 s with 8. Use `num_workers=0` for small
  interactive subsets and persistent workers (`persistent_workers=True`) for
  full training, where start-up is paid once.

### Weakly mixed batches with `shuffle_buffer="io"`

`"io"` only guarantees that one decoded block does not land whole in a batch.
On a real multi-store epoch it resolved to a buffer of 48 and measurably hurt
training. Use the default (2,048), `"auto"` (never below 2,048), or an
explicit count, and check `ds.plan(batch_size=...)`.

### The DataLoader hangs

- **Hang at start-up or at an epoch boundary** with `fork`: this is the
  libomp/BLAS lock inherited by a forked worker. See *Multiprocessing safety*
  above. Use `OMP_NUM_THREADS=1` and `multiprocessing_context="forkserver"`.
- **Rare hang under extreme node load**: we saw it once with a load average
  around 170. The main process waited for a batch while every worker sat idle
  on its index queue, and no zarrpatch frame was on any stack. Pass
  `DataLoader(timeout=...)` so a hang becomes an exception. If it recurs, take
  a `py-spy dump` of the main process and of every worker.

### Benchmarks that disagree between runs

On shared parallel filesystems (GPFS, Lustre), "cold" reads cannot be forced
without admin rights. `posix_fadvise` does not evict GPFS's pagepool, so
first-epoch numbers are best-effort, and run-to-run noise on a busy node
reached ±30%. Compare warm epochs, and run the alternatives you are comparing
in the same session.

## Layout

```
src/zarrpatch/
  readers.py     zarr (v2/v3, ±shards, .zarr.zip), TensorStore, netCDF,
                 kerchunk and MultiCubeReader backends, one interface
  samplers.py    RandomSampler / GridSampler / ListSampler / MaskedSampler /
                 WeightedSampler, coords_to_indices
  shuffle.py     streaming shuffle buffer
  dataset.py     ZarrPatchDataset (torch IterableDataset, block-grouped reads,
                 equal-count rank/worker split)
  planning.py    ExecutionPlan / analyze_layout — read-free epoch plan,
                 decode amplification, batch-mixing replay
  materialize.py materialize() / HDF5PatchDataset — frozen snapshots
  profiling.py   process-safe JSONL tracing and startup/I/O summaries
  reconstruct.py Reconstructor — weighted-overlap rebuild from patch predictions
  guidelines.py  recommend_layout() — chunk/shard advice per sampling strategy
  bench.py       benchmark(), print_results(), results_to_markdown()
tests/           pytest suite incl. exactness vs direct slicing, multi-worker,
                 equal DDP ranks, persistent workers, masked/weighted sampling,
                 dim slices, zip stores, multi-cube reads, planning replay,
                 reconstruction round-trips
docs/            loading-primer.md: where loading time goes, formats, strategies
benchmarks/      run_bench.py + results.md, ecosystem_bench.py +
                 results_ecosystem.md, time_breakdown.py +
                 results_breakdown.md
```

## License

Apache-2.0, see [`LICENSE`](LICENSE).
