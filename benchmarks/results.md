# zarrpatch benchmark — medium cube

dims {'time': 100, 'lat': 600, 'lon': 600}, patch {'time': 5, 'lat': 64, 'lon': 64}, 2000 random patches, batch 32, workers 0, buffer 2048

chunks {'time': 10, 'lat': 100, 'lon': 100}, shards {'time': 50, 'lat': 300, 'lon': 300}. `items` counts patches; speedup is vs `netcdf / naive`.

| loader | items | wall [s] | first item [s] | items/s | MiB/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| netcdf / naive | 2000 | 3.13 | 0.062 | 639.0 | 49.9 | 1.00x |
| zarr v2 / naive | 2000 | 1.77 | 0.035 | 1132.5 | 88.5 | 1.77x |
| zarr v3 / naive | 2000 | 2.08 | 0.042 | 959.4 | 75.0 | 1.50x |
| zarr v3 sharded / naive | 2000 | 5.22 | 0.108 | 383.0 | 29.9 | 0.60x |
| zarr v3 / grouped, no buffer | 2000 | 0.60 | 0.011 | 3356.1 | 262.2 | 5.25x |
| zarr v3 sharded / grouped, no buffer | 2000 | 0.27 | 0.015 | 7456.2 | 582.5 | 11.67x |
| zarr v3 sharded / grouped + buffer | 2000 | 0.26 | 0.250 | 7671.8 | 599.4 | 12.01x |
| zarr v3 sharded / grouped + buffer + prefetch | 2000 | 0.25 | 0.239 | 8041.3 | 628.2 | 12.58x |
