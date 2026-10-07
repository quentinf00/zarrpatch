| scenario | store | loader | workers | epoch 1 /s | epoch 1 first batch [s] | warm /s | warm CPU [s] | warm /CPU-s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| random | unsharded | zarrpatch | 0 | 1615 | 1.64 | 1934 | 20.3 | 983 |
| random | unsharded | zarrpatch group_block | 0 | 4110 | 0.62 | 4433 | 8.2 | 2436 |
| random | unsharded | zarrpatch on TS 1t | 0 | 7450 | 0.41 | 6880 | 6.3 | 3165 |
| random | sharded | zarrpatch | 0 | 3206 | 0.70 | 4327 | 8.3 | 2401 |
| random | sharded | zarrpatch on TS 1t | 0 | 8218 | 0.32 | 8337 | 5.0 | 3984 |
| random | unsharded | zarrpatch | 8 | 2350 | 7.21 | 7345 | 23.2 | 864 |
| random | unsharded | zarrpatch group_block | 8 | 3820 | 4.56 | 15788 | 10.8 | 1850 |
| random | unsharded | zarrpatch on TS 1t | 8 | 4385 | 4.12 | 21490 | 7.7 | 2601 |
| random | sharded | zarrpatch | 8 | 4029 | 4.32 | 14659 | 11.1 | 1797 |
| random | sharded | zarrpatch on TS 1t | 8 | 5972 | 2.87 | 24844 | 6.7 | 2981 |
| grid | unsharded | zarrpatch | 0 | 2021 | 1.12 | 2190 | 11.2 | 1161 |
| grid | unsharded | zarrpatch group_block | 0 | 4012 | 0.53 | 4593 | 5.4 | 2418 |
| grid | unsharded | zarrpatch on TS 1t | 0 | 6500 | 0.30 | 6950 | 4.1 | 3169 |
| grid | sharded | zarrpatch | 0 | 3592 | 0.58 | 4500 | 5.5 | 2335 |
| grid | sharded | zarrpatch on TS 1t | 0 | 6637 | 0.36 | 7534 | 3.7 | 3493 |
| grid | unsharded | zarrpatch | 8 | 978 | 12.98 | 9256 | 13.3 | 976 |
| grid | unsharded | zarrpatch group_block | 8 | 1092 | 11.70 | 15071 | 7.2 | 1805 |
| grid | unsharded | zarrpatch on TS 1t | 8 | 1136 | 11.13 | 24325 | 5.0 | 2592 |
| grid | sharded | zarrpatch | 8 | 1100 | 11.55 | 13781 | 8.3 | 1567 |
| grid | sharded | zarrpatch on TS 1t | 8 | 1245 | 10.19 | 20980 | 5.1 | 2541 |
