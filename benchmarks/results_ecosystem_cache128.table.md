| scenario | store | loader | workers | epoch 1 /s | epoch 1 first batch [s] | warm /s | warm CPU [s] | warm /CPU-s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| random | unsharded | tensorstore+cache 128M | 0 | 477 | 1.56 | 2072 | 24.4 | 819 |
| random | sharded | tensorstore+cache 128M | 0 | 1188 | 2.02 | 1952 | 30.0 | 666 |
| random | unsharded | tensorstore+cache 128M | 8 | 757 | 22.27 | 6992 | 31.4 | 637 |
| random | sharded | tensorstore+cache 128M | 8 | 2462 | 4.94 | 6749 | 31.7 | 630 |
| grid | unsharded | tensorstore+cache 128M | 0 | 1812 | 0.07 | 2010 | 13.7 | 946 |
| grid | sharded | tensorstore+cache 128M | 0 | 1905 | 0.17 | 2014 | 15.0 | 863 |
| grid | unsharded | tensorstore+cache 128M | 8 | 1692 | 6.21 | 9968 | 14.4 | 897 |
| grid | sharded | tensorstore+cache 128M | 8 | 2628 | 3.55 | 9310 | 15.1 | 860 |
