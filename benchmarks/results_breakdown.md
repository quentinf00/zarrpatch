# Where the time goes: one warm epoch, stage by stage

Synthetic `medium` cube {'time': 100, 'lat': 600, 'lon': 600} float32, zarr v3, chunks {'time': 10, 'lat': 100, 'lon': 100}, shards {'time': 50, 'lat': 300, 'lon': 300}; 4000 random {'time': 5, 'lat': 64, 'lon': 64} patches (0.08 MiB each), batch 32. One process, read-ahead off (`prefetch=0`), warm page cache, `torch.set_num_threads(1)`. 64 CPUs, Python 3.12.13, torch 2.12.1. Produced by `benchmarks/time_breakdown.py`.

## Stages (seconds per epoch; % of the epoch excluding open)

| configuration | open | plan | read (I/O + decode + zarr) | slice + shuffle | collate | first patch after | groups | read amplification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk groups, zstd | 0.18 | 0.01 (0%) | 1.79 (63%) | 1.02 (36%) | 0.03 (1%) | 1.35 | 357 | 1.0× |
| chunk groups, uncompressed | 0.01 | 0.01 (0%) | 1.30 (57%) | 0.95 (41%) | 0.03 (1%) | 1.20 | 357 | 1.0× |
| shard groups, zstd | 0.01 | 0.01 (1%) | 0.46 (31%) | 0.96 (65%) | 0.05 (3%) | 0.76 | 8 | 0.6× |
| group_block 3x chunk, zstd | 0.01 | 0.01 (1%) | 0.45 (33%) | 0.81 (59%) | 0.09 (7%) | 0.60 | 16 | 0.6× |

Decompression (zstd minus uncompressed reads, chunk groups): **0.49 s**, 27% of the read stage.

## What DataLoader workers add (samples/s, warm epoch)

| workers | zarrpatch (shard groups) | in-memory control (collate + transfer ceiling) |
|---:|---:|---:|
| 0 | 2,778 | 64,216 |
| 4 | 5,739 | 19,120 |
| 8 | 8,068 | 25,929 |
