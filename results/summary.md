# Experiment 3 summary: regression or architecture?

Device: Xiaomi 23117RA68G (Redmi 13), Helio G99 (MT6789), Android 13. LFM2-VL-450M Q8_0 + mmproj Q8_0, llama.cpp CPU, `-t 4` unless stated.
Current build: `a691b2345844ce1b430201fc5fd404ce062e05f2` (2026-11-08). Old build: tag `b10516` (`b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9`, 2026-08-19), 1689 commits earlier. Same CMake flags.
Temperature: battery (`dumpsys battery`) and CPU (`dumpsys hardware_properties`, readable without root) recorded before every run in the CSVs. Runs waited if battery rose 3 C or CPU 10 C above the start; no waiting was needed (battery 37-38 C, CPU 49-51 C).

## Verdict
**Not a regression on our phone.** The encoder on the current build is 0.97-0.99x the time of b10516 (same speed within noise). The slow encoder is the architecture story: the encoder runs a non-repacked int8 kernel (tinyBLAS), the LM runs the repacked `q8_0_4x4` kernel.

## 0. Experiment 1, rescored (N=16 fully measured images, stopped early from planned 30)
`strict`: number equals GT (after removing `$`, a trailing `.00`/`,00`, and treating a separator before exactly three digits as a thousands separator).
`digits`: digit string equals the GT digit string after stripping trailing zeros. More lenient than strict; allows false matches (e.g. 127 vs 1270).

| condition | tokens | encoder ms (med [p25,p75]) | image prefill ms | text prefill ms | TTFT ms (med [p25,p75]) | strict | digits |
|---|---|---|---|---|---|---|---|
| A full shelf (tiled) | 242 | 5063 [4499,14143] | 1296 | 220 | 6536 [6013,18564] | 5/16 | 7/16 |
| A2 full shelf (1 tile) | 250 | 5105 [4920,5222] | 1273 | 212 | 6565 [6380,6736] | 6/16 | 9/16 |
| B shelf down to ~70 tok | 74 | 1429 [1314,1492] | 402 | 196 | 2032 [1898,2135] | 4/16 | 5/16 |
| C tag + context | 81 | 1601 [1429,2171] | 456 | 202 | 2261 [2039,3006] | 8/16 | 10/16 |
| D tight tag crop | 72 | 1419 [1374,1494] | 408 | 197 | 2022 [1952,2114] | 10/16 | 10/16 |
| E tight crop, 20 tok | 20 | 479 [448,516] | 113 | 189 | 784 [745,830] | 5/16 | 9/16 |

Paired B vs D (strict): only D correct 7, only B correct 1, both/neither 8; two-sided exact sign test p = 0.070 (suggestive, not conclusive at N=16).
Plots for a post: `results/post/` (`time_by_phase.png` is the main one; tiled A is shown separately in `full_shelf_tiled_ttft.png`).
Encoder cost: about 20 ms per image token, R^2 = 0.99.

## 1. Model sizes (GGUF metadata) and achieved throughput
- mmproj: 12 layers, width 768, MLP 3072, patch 16, pixel-shuffle scale 2 (4 patches per token), 96.3M params.
- LM (`lfm2`): 16 layers (attention layers have KV heads, the rest are short-conv, `l_cache` 3), width 1024, MLP 4608, vocab 65536; 354.5M params total, 67.1M in the token embedding, 287.4M without it.
- Encoder FLOP = L*(2N(4d^2 + 2 d d_mlp) + 4N^2 d), N = tokens*4. LM FLOP = 2*(287.4M)*(image tokens). Short-conv layers are approximated as dense; patch embedding and projector are ignored.

| cond | tokens | patches N | encoder GFLOP | encoder ms | encoder GOPS | LM GOP | image prefill ms | LM GOPS |
|---|---|---|---|---|---|---|---|---|
| A2 | 250 | 1000 | 207 | 5105 | 40.0 | 144 | 1273 | 110.6 |
| B | 74 | 294 | 53 | 1429 | 37.3 | 42 | 402 | 105.4 |
| C | 81 | 324 | 59 | 1601 | 38.0 | 47 | 456 | 104.3 |
| D | 72 | 288 | 52 | 1419 | 37.0 | 41 | 408 | 102.9 |
| E | 20 | 82 | 14 | 479 | 29.3 | 12 | 113 | 102.9 |

The LM prefill runs at about 2.7x the throughput of the encoder (103-111 vs 37-40 GOPS). Against the rough int8 dotprod peak estimate of 280 GOPS for 2 A76 cores that is about 38% vs 14% (the estimate is not a measurement).

## 2. Current build vs b10516
Issue ggml-org/llama.cpp#27649: **closed, `not_planned`**, closed by the stale bot on 2026-10-08, one comment (the bot), no fix commit referenced. It was reported on a different model (OvisOCR2, Qwen3.5-based mmproj F16) and chip (Snapdragon 8 Elite, Termux), commit `a130532` (2026-08-23).
b10516 loads LFM2-VL and the mmproj without problems and gives identical image-token counts and answers.
Method: conditions A2 (image 10) and D (image 09), the median-TTFT image of each condition in experiment 1; 5 repeats per build, builds alternated (order swapped every repetition), `-t 4`.

| cond | phase | current med [min,max] ms | b10516 med [min,max] ms | current / old |
|---|---|---|---|---|
| A2 | encoder | 5082 [4850,5464] | 5152 [4960,5805] | 0.99 |
| A2 | image prefill | 1254 [1239,1276] | 1307 [1270,1314] | 0.96 |
| A2 | text prefill | 224 [215,238] | 232 [225,252] | 0.96 |
| D | encoder | 1675 [1429,1951] | 1733 [1459,1877] | 0.97 |
| D | image prefill | 411 [410,419] | 421 [411,424] | 0.98 |
| D | text prefill | 205 [200,206] | 208 [204,211] | 0.99 |

Answers match on both builds (A2 1180, D 1290). Threshold from the plan (old at least 1.3x faster) is not reached; no bisection needed.

## 3. simpleperf attribution (current build, A2 image 10, `-t 4`, dwarf call graph)
Worker-thread stacks start at `pthread_start`, so the call graph does not link them to `clip_image_batch_encode` / `llama_decode`. The encoder and the LM have separate threadpools, so attribution was done per thread:
- Threads 4482-4484 (encoder pool) run only `tinyBLAS_Q0_ARM<block_q8_0>::gemm` (about 9.4-9.6% of all samples per thread) and `flash_attn_ext` (about 3.5-3.9% per thread).
- Threads 4479-4481 (LM pool) run only `ggml_gemm_q8_0_4x4_q8_0` and `ggml_gemv_q8_0_4x4_q8_0` (about 3.4-3.6% per thread for gemm).
- The main thread (4473) works for both.
Confirmed: **tinyBLAS = encoder, `q8_0_4x4` = LM**. Whole-profile self time: tinyBLAS gemm 38.0%, `flash_attn_ext` 15.2% (mostly encoder), `ggml_gemm_q8_0_4x4` 13.8%, `ggml_graph_compute_thread` 10.3%, `ggml_barrier` 0.8%. Barrier wait inside `ggml_graph_compute_thread` could not be separated.
Core distribution (simpleperf has no per-CPU key; sampled `ps -T` PSR every 100 ms instead, role assignment by thread id order is approximate): threads spread over all 8 cores; the encoder pool spent about 37% of its running samples on cpu6-7 (A76) and the rest on the A55 cores.
A profile of b10516 was not taken because the builds do not differ.

## 4. Thread pinning (condition D, image 09, 5 repeats interleaved)
| config | encoder ms med [min,max] | image prefill ms | text prefill ms | TTFT ms |
|---|---|---|---|---|
| `-t 4` unpinned | 1485 [1437,1669] | 413 | 203 | 2097 |
| `taskset c0` + `-t 2` (cpu6-7) | 1275 [1268,1313] | 415 | 206 | 1895 |
| `-t 8` unpinned | 1521 [1485,2130] | 412 | 342 | 2329 |
Two threads pinned to the big cores are about 14% faster on the encoder and about 10% on TTFT than `-t 4` unpinned, and far more stable. `-t 8` is not better (text prefill slows). Answers identical (1290). Pinning applies to the whole process (`taskset`), so the encoder pool is pinned too; the `--cpu-mask` flag was not tested.

## 5. Mobile GPU (Mali-G57 MC2)
Vulkan segfault root causes found and fixed locally (details in `docs/vulkan-mali-findings.md`, patch `patches/vulkan-android-1.1-mali.diff`): the Mali driver is Vulkan 1.1.177 with the needed features only as extensions, while `ggml-vulkan` uses 1.2 core entry points (buffer device address, timeline semaphores) and an unusable `vkGetDeviceQueue2` queue. With the patch Vulkan runs and answers match the CPU.
It is not faster: tight crop (D), TTFT median CPU 1.9-2.3 s, Vulkan LM-only 9.3 s, Vulkan all 14.2 s (table: `results/exp3_gpu.csv`). Cause: the GPU delivers only about 7-11 GOPS in `ggml-vulkan` (LM prefill about 8 GOPS vs 103-111 on the CPU; encoder 7-11 vs 37-40), confirmed by a timestamped `simpleperf` profile (CPU waiting on the fence while the GPU computes). Separately, shaders are compiled lazily without a persistent cache: 7-10 s per process, mostly in warm-up, three fused rms-norm kernels at about 2 s each.
Correction: an earlier note that warm image prefill on the GPU takes 27-39 ms was a measurement artefact (the image decode returns after submission, the wait lands on the next text decode).
OpenCL: the backend rejects Mali by design (Adreno/Intel only); an experimental allowlist change still fails to compile the kernels (they assume Adreno subgroup features). Not usable.

## Cold start vs warm steady state (long-lived `llama-server`, tight crop D and one-tile frame A2)
Same builds, one process per configuration, requests with `cache_prompt: false`, 2 warm-up requests, then 5 measured; TTFT = server `prompt_ms` (image encode + image tokens + prompt text). Battery 38 C, CPU 48-51 C during the runs. Raw: `results/exp4_warm.csv`, script `scripts/exp4_warm.py`.

| config | server ready (s) | D: first request ms | D: warm median [min,max] ms | A2: first request ms | A2: warm median [min,max] ms |
|---|---|---|---|---|---|
| CPU `-t 4` | 2.1 | 2145 | 2452 [2395,2483] | 6539 | 7805 [7702,8208] |
| CPU 2 big cores | 2.1 | 2272 | 2279 [2247,2293] | 7995 | 8038 [8011,8055] |
| Vulkan, LM on GPU, encoder CPU | 10.7 | 9018 | 7488 [7290,7644] | 17154 | 17591 [17354,17884] |
| Vulkan, all on GPU | 11.3 | 14941 | 13291 [13163,13339] | 31224 | 31245 [31119,31541] |

- Caching helps the GPU only a little: the first request pays about 1.5 s of extra compilation, the warm request is still 7.5 s (LM on GPU) and 13.3 s (all on GPU) vs 2.3-2.5 s on the CPU. So the GPU slowness is steady-state compute (about 7-11 GOPS), not cold start.
- Cold start is a separate cost: the Vulkan server needs 10.7-11.3 s to become ready (shader compilation during warm-up) vs 2.1 s on the CPU.
- The CPU server numbers are 10-20% above the CLI (D 2.3-2.5 s vs 1.9-2.3 s, A2 7.8-8.0 s vs 6.6 s): base64/JPEG handling on the device, HTTP, and a CPU already at about 50 C. Compare configurations within one table, not across tools. Answers are identical on every request and configuration (D 1290, A2 1180).

## Deviations and caveats
- N=16 of the planned 30 images; data are display warps from a private dataset, not original photos.
- Condition B and A2 use host-pre-resized frames (`--image-max-tokens` does not limit tiling).
- Single run per condition and image in experiment 1; the current-vs-old test uses one image per condition.
- GOPS use the formula above, short-conv layers approximated as dense.
- Raw data: `exp1_runs.csv`, `exp3_regression.csv`, `exp3_pinning.csv`; tables `exp3_table.md`, `exp3_gops.md`.
