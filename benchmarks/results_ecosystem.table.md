| scenario | store | loader | workers | epoch 1 /s | epoch 1 first batch [s] | warm /s | warm CPU [s] | warm /CPU-s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| random | unsharded | xarray isel | 0 | 340 | 1.35 | 864 | 45.8 | 437 |
| random | unsharded | zarr per-patch | 0 | 717 | 0.07 | 745 | 49.5 | 404 |
| random | unsharded | tensorstore | 0 | 1825 | 0.63 | 2133 | 24.8 | 808 |
| random | unsharded | tensorstore+cache | 0 | 5084 | 0.03 | 5362 | 5.9 | 3407 |
| random | unsharded | tensorstore batch | 0 | 8730 | 0.02 | 8591 | 25.2 | 793 |
| random | unsharded | tensorstore batch 1t | 0 | 1231 | 0.04 | 1238 | 25.6 | 780 |
| random | unsharded | zarrpatch | 0 | 2497 | 0.79 | 2586 | 15.5 | 1293 |
| random | unsharded | zarrpatch group_block | 0 | 5851 | 0.43 | 6163 | 6.1 | 3284 |
| random | unsharded | zarrpatch on TS 1t | 0 | 10731 | 0.23 | 10013 | 4.5 | 4454 |
| random | sharded | xarray isel | 0 | 442 | 1.94 | 531 | 76.2 | 262 |
| random | sharded | zarr per-patch | 0 | 477 | 0.16 | 492 | 79.6 | 251 |
| random | sharded | tensorstore | 0 | 2443 | 0.04 | 2500 | 23.5 | 850 |
| random | sharded | tensorstore+cache | 0 | 7616 | 0.03 | 8338 | 4.3 | 4651 |
| random | sharded | tensorstore batch | 0 | 8449 | 0.02 | 8490 | 24.5 | 818 |
| random | sharded | tensorstore batch 1t | 0 | 1210 | 0.04 | 1229 | 25.2 | 794 |
| random | sharded | zarrpatch | 0 | 6045 | 0.41 | 6147 | 6.1 | 3273 |
| random | sharded | zarrpatch on TS 1t | 0 | 8060 | 0.35 | 11414 | 3.6 | 5495 |
| random | unsharded | xarray isel | 8 | 647 | 22.99 | 3691 | 52.2 | 383 |
| random | unsharded | zarr per-patch | 8 | 1823 | 4.92 | 3022 | 60.5 | 331 |
| random | unsharded | tensorstore | 8 | 3462 | 3.50 | 9045 | 27.9 | 717 |
| random | unsharded | tensorstore+cache | 8 | 4786 | 2.82 | 25846 | 7.3 | 2751 |
| random | unsharded | tensorstore batch | 8 | 4605 | 2.64 | 11854 | 25.1 | 796 |
| random | unsharded | tensorstore batch 1t | 8 | 4024 | 2.68 | 8733 | 24.0 | 832 |
| random | unsharded | zarrpatch | 8 | 3895 | 4.40 | 9544 | 20.4 | 982 |
| random | unsharded | zarrpatch group_block | 8 | 5159 | 3.51 | 19425 | 8.8 | 2281 |
| random | unsharded | zarrpatch on TS 1t | 8 | 5601 | 3.25 | 30591 | 5.9 | 3373 |
| random | sharded | xarray isel | 8 | 1439 | 3.73 | 2061 | 101.6 | 197 |
| random | sharded | zarr per-patch | 8 | 1397 | 3.59 | 1882 | 107.3 | 186 |
| random | sharded | tensorstore | 8 | 3487 | 3.39 | 8483 | 29.2 | 686 |
| random | sharded | tensorstore+cache | 8 | 5544 | 2.39 | 31390 | 6.0 | 3350 |
| random | sharded | tensorstore batch | 8 | 5703 | 1.87 | 12136 | 25.2 | 795 |
| random | sharded | tensorstore batch 1t | 8 | 4791 | 1.82 | 8498 | 24.0 | 832 |
| random | sharded | zarrpatch | 8 | 6096 | 2.82 | 19572 | 9.1 | 2188 |
| random | sharded | zarrpatch on TS 1t | 8 | 6661 | 2.66 | 31338 | 5.5 | 3604 |
| grid | unsharded | xarray isel | 0 | 279 | 0.73 | 1046 | 22.1 | 586 |
| grid | unsharded | zarr per-patch | 0 | 810 | 2.94 | 846 | 25.5 | 509 |
| grid | unsharded | tensorstore | 0 | 2152 | 0.04 | 2254 | 11.8 | 1098 |
| grid | unsharded | tensorstore+cache | 0 | 5195 | 0.02 | 7728 | 2.5 | 5205 |
| grid | unsharded | tensorstore batch | 0 | 10099 | 0.01 | 10625 | 12.0 | 1084 |
| grid | unsharded | tensorstore batch 1t | 0 | 1689 | 0.02 | 1697 | 12.1 | 1070 |
| grid | unsharded | xbatcher | 0 | 849 | 0.99 | 900 | 24.6 | 527 |
| grid | unsharded | zarrpatch | 0 | 2618 | 0.86 | 2674 | 8.9 | 1453 |
| grid | unsharded | zarrpatch group_block | 0 | 5976 | 0.38 | 5986 | 4.2 | 3123 |
| grid | unsharded | zarrpatch on TS 1t | 0 | 9841 | 0.23 | 9355 | 3.0 | 4393 |
| grid | sharded | xarray isel | 0 | 426 | 2.09 | 587 | 41.3 | 314 |
| grid | sharded | zarr per-patch | 0 | 527 | 3.85 | 520 | 45.0 | 288 |
| grid | sharded | tensorstore | 0 | 2639 | 0.02 | 2689 | 11.5 | 1131 |
| grid | sharded | tensorstore+cache | 0 | 6865 | 0.02 | 9219 | 2.3 | 5562 |
| grid | sharded | tensorstore batch | 0 | 10512 | 0.01 | 10449 | 12.3 | 1057 |
| grid | sharded | tensorstore batch 1t | 0 | 1661 | 0.02 | 1677 | 12.1 | 1069 |
| grid | sharded | xbatcher | 0 | 517 | 0.16 | 531 | 44.1 | 294 |
| grid | sharded | zarrpatch | 0 | 5706 | 0.40 | 5792 | 4.2 | 3064 |
| grid | sharded | zarrpatch on TS 1t | 0 | 9924 | 0.24 | 9271 | 2.8 | 4580 |
| grid | unsharded | xarray isel | 8 | 531 | 18.35 | 4874 | 26.7 | 486 |
| grid | unsharded | zarr per-patch | 8 | 1040 | 12.31 | 3700 | 29.6 | 437 |
| grid | unsharded | tensorstore | 8 | 2608 | 3.79 | 11621 | 12.9 | 1002 |
| grid | unsharded | tensorstore+cache | 8 | 3184 | 3.11 | 33768 | 3.5 | 3661 |
| grid | unsharded | tensorstore batch | 8 | 3687 | 2.69 | 15622 | 12.2 | 1060 |
| grid | unsharded | tensorstore batch 1t | 8 | 3398 | 2.72 | 11550 | 11.8 | 1100 |
| grid | unsharded | xbatcher | 8 | 1900 | 3.93 | 4381 | 28.4 | 457 |
| grid | unsharded | zarrpatch | 8 | 1209 | 10.49 | 10818 | 11.3 | 1147 |
| grid | unsharded | zarrpatch group_block | 8 | 1229 | 10.36 | 18529 | 6.3 | 2064 |
| grid | unsharded | zarrpatch on TS 1t | 8 | 1349 | 9.43 | 26161 | 4.3 | 3007 |
| grid | sharded | xarray isel | 8 | 1372 | 3.85 | 2287 | 55.7 | 233 |
| grid | sharded | zarr per-patch | 8 | 811 | 15.48 | 2079 | 58.4 | 222 |
| grid | sharded | tensorstore | 8 | 3685 | 2.35 | 10870 | 14.0 | 926 |
| grid | sharded | tensorstore+cache | 8 | 4721 | 1.89 | 37627 | 3.2 | 4012 |
| grid | sharded | tensorstore batch | 8 | 4669 | 1.93 | 15554 | 12.4 | 1049 |
| grid | sharded | tensorstore batch 1t | 8 | 4309 | 1.88 | 11059 | 12.1 | 1068 |
| grid | sharded | xbatcher | 8 | 1477 | 2.88 | 2222 | 58.0 | 223 |
| grid | sharded | zarrpatch | 8 | 1212 | 10.52 | 16405 | 6.7 | 1934 |
| grid | sharded | zarrpatch on TS 1t | 8 | 1434 | 8.83 | 26104 | 4.2 | 3108 |
