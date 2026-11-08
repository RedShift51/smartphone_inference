| condition | N | tokens | encoder ms (med [p25,p75]) | image prefill ms | text prefill ms | TTFT ms (med [p25,p75]) | strict | digits |
|---|---|---|---|---|---|---|---|---|
| A full shelf (tiled) | 16 | 242 | 5063 [4499,14143] | 1296 | 220 | 6536 [6013,18564] | 5/16 | 7/16 |
| A2 full shelf (1 tile) | 16 | 250 | 5105 [4920,5222] | 1273 | 212 | 6565 [6380,6736] | 6/16 | 9/16 |
| B shelf ↓ to ~70 tok | 16 | 74 | 1429 [1314,1492] | 402 | 196 | 2032 [1898,2135] | 4/16 | 5/16 |
| C tag + context | 16 | 81 | 1601 [1429,2171] | 456 | 202 | 2261 [2039,3006] | 8/16 | 10/16 |
| D tight tag crop | 16 | 72 | 1419 [1374,1494] | 408 | 197 | 2022 [1952,2114] | 10/16 | 10/16 |
| E tight crop, 20 tok | 16 | 20 | 479 [448,516] | 113 | 189 | 784 [745,830] | 5/16 | 9/16 |

Paired B vs D (strict, N=16): only D correct 7, only B correct 1, both/neither 8; two-sided exact sign test p = 0.070.
`digits` = digit string of the answer equals the digit string of GT after stripping trailing zeros; it is more lenient than `strict` and allows false matches (e.g. 127 vs 1270).
