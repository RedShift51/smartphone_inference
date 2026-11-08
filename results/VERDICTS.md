# All verdicts (LFM2-VL-450M on Redmi 13, Helio G99)

Setup: Redmi 13 (Helio G99 / MT6789, 2x A76 2.2 GHz + 6x A55 2.0 GHz, Mali-G57 MC2, 12 GB), Android 13. LFM2-VL-450M Q8_0 (LM and mmproj), llama.cpp `a691b2345844ce1b430201fc5fd404ce062e05f2` (2026-11-08), `llama-mtmd-cli`, CPU unless stated. Task: read a 4-digit price from a price tag. Battery and CPU temperature were recorded before every run. Nothing here was committed, pushed or published. Details and raw data: `summary.md`, `final_report.md`, `docs/`, CSV files in `results/`.

## 1. Where the time goes
- **VERDICT: the vision encoder is 75-80% of TTFT; the LM and the prompt text are small.** Tight crop median: encoder 1.4 s, image tokens in the LM 0.4 s, prompt text 0.2 s, TTFT 2.0 s.
- Cost per image token: encoder about 20 ms (R^2 = 0.99); LM prefill about 5.5 ms.

## 2. Crop vs full frame
- **VERDICT: at the same token budget and the same latency a tight tag crop reads the price much better than a downscaled full frame.** Tight crop 72 tokens, 2.0 s: **10/16** exact; full shelf squeezed to about 74 tokens, 2.0 s: **4/16**. Paired sign test 7 vs 1, **p = 0.07**: suggestive, not conclusive at N = 16. Crop with context (about 81 tokens): 8/16. One-tile full frame (about 250 tokens, 6.6 s): 6/16. Default tiled full frame: 5/16.
- Lenient metric (digit sequence must match, trailing zeros stripped; allows false matches): D 10, C 10, A2 9, E 9, A 7, B 5 (of 16).
- **VERDICT: the model is weak at this task in general (450M): even the best condition misses 6 of 16 tags.** Latency numbers do not depend on correctness.

## 3. Token budget and tiling
- **VERDICT: going lower in tokens trades accuracy for speed.** Tight crop at 20 tokens: TTFT 0.8 s, exact 5/16 (lenient 9/16).
- **VERDICT: the default full-frame path tiles large frames into 512x512 tiles plus an overview, so cost explodes.** Median TTFT 6.5 s, tail up to 49 s (1776 tokens). `--image-max-tokens` does **not** limit tiling (it only shrinks the overview).
- Small crops are not upscaled to exactly 64 tokens (aspect ratio is kept): tight crop gives 70-78 tokens; `--image-min-tokens 16 --image-max-tokens 16` gives 20-21.
- **VERDICT (honest TTFT): amortising the price-tag detector (805 ms per frame on CPU, about 10.3 tags per frame, about 78 ms per tag) puts the tight crop at about 2.1 s.**

## 4. Regression or architecture
- **VERDICT: not a regression on this phone and model.** Current build vs tag `b10516` (2026-08-19, 1689 commits earlier): encoder time ratio 0.97-0.99 (5 alternating repeats, identical answers). Issue ggml-org/llama.cpp#27649 is closed as `not_planned` by the stale bot with no fix, and was reported on another model and chip. Threshold from the plan (old >= 1.3x faster) not reached; no bisection done.
- **VERDICT: the architecture explanation holds.** The encoder runs a generic int8 kernel (`tinyBLAS_Q0_ARM`) at 30-40 GOPS achieved; the LM runs the repacked `q8_0_4x4` kernel at 103-111 GOPS (2.7x). Confirmed by per-thread attribution in simpleperf (encoder pool threads run only tinyBLAS and flash attention, LM pool threads only the repacked gemm/gemv).
- Code reading (G1 holds): mmproj weights go into the default CPU buffer type (`clip.cpp:4815`), the LM uses extra buffer types (`llama-model.cpp:1262`). Correction to G2: on this phone Q8_0 is also repacked for the LM (`q8_0_4x4`), not only Q4_0 (`q4_0_4x4`) and Q4_K (`q4_K_8x4`); 4x8/8x8 variants need i8mm and are not used.
- Model sizes: mmproj 96.3M params (12 layers, width 768, MLP 3072, patch 16, pixel shuffle 2); LM 354.5M total, 287.4M without the 67.1M token embedding (16 layers, width 1024, MLP 4608, vocab 65536, short-conv + attention).

## 5. Encoder repack fix (local experiment)
- **VERDICT: placing the mmproj weights in the CPU repack buffer type removes most of the encoder penalty: the encoder is 1.6-1.8x faster and TTFT 1.4-1.45x faster, with identical answers.** Patch: 2D Q8_0 weights (74 tensors) go into the first extra buffer type, enabled by `CLIP_REPACK=1` (`patches/mtmd-encoder-repack-experiment.diff`, 24 lines, local worktree only).
- Precise interleaved A/B (10 repeats, order shuffled per round, battery 40 C, CPU 51 C), medians:

| cond | config | encoder ms | TTFT ms |
|---|---|---|---|
| D | `-t 4` baseline | 1663 [1623,1702] | 2389 |
| D | `-t 4` repack | 933 [928,939] | 1663 |
| D | 2 big cores baseline | 1503 | 2244 |
| D | 2 big cores repack | 869 | 1606 |
| A2 | `-t 4` baseline | 6090 [5963,6120] | 7899 |
| A2 | `-t 4` repack | 3742 [3728,3761] | 5596 |
| A2 | 2 big cores baseline | 6135 | 7988 |
| A2 | 2 big cores repack | 3910 | 5765 |

  Paired median speedups (bootstrap 95% CI): D encoder 1.80x [1.73,1.86], TTFT 1.45x [1.41,1.48]; A2 encoder 1.62x [1.60,1.64], TTFT 1.41x [1.40,1.43]; pinned: D 1.72x, A2 1.57x. Achieved encoder throughput about 30 -> 50-54 GOPS (still below the LM's 103-111).
- Answers identical with and without repack on all 16 tight crops and on every A2 run; accuracy on the other conditions was not re-measured.
- Caveat: this session ran hotter (battery 40 C, CPU 51 C), so absolute times are 10-20% higher than earlier runs; the paired ratios are the result. The first quick A/B (5 repeats) gave 1.7-2.0x for the encoder, consistent.
- Not upstreamed. This is a local experiment, a proper change would need per-tensor handling in `clip.cpp` and a safe fallback in the repack buffer for tensors without traits.

## 6. Threads and core pinning
- **VERDICT: more threads do not help.** Condition A, `-t 2/4/6`: median prompt eval 21.5 / 21.0 / 21.5 s (differences below thermal drift). `-t 8` is not better (text prefill slows from 203 to 342 ms).
- **VERDICT: 2 threads pinned to the two big cores (`taskset c0`) are better and steadier than `-t 4` unpinned** on the baseline build: encoder 1275 vs 1485 ms (-14%), TTFT about -10% (condition D). With the repack patch the gain mostly shrinks (D 869 vs 933 ms) and disappears for A2 (3910 vs 3742 ms), so pinning matters most when the encoder is the slow generic kernel.
- Unpinned threads spent only about 37% of their running time on the big cores (sampled by `ps -T`, thread roles approximate). `--cpu-mask` was not tested; by the code it does not pin the encoder's own threadpool.

## 7. Mobile GPU
- **VERDICT (Vulkan): the segfault is a Vulkan-version mismatch, fixable locally.** The Mali-G57 driver is Vulkan 1.1.177 with `VK_KHR_buffer_device_address` and `VK_KHR_timeline_semaphore` only as extensions, while `ggml-vulkan` only checks the instance version and calls 1.2 core entry points through a dispatcher initialised from the instance. Three crashes, in order: `vkGetBufferDeviceAddress` NULL (`ggml-vulkan-buffers.cpp:202`), `vkQueueSubmit` with a queue from `vkGetDeviceQueue2` (`ggml-vulkan.cpp:1050`), timeline-semaphore calls NULL (`ggml-vulkan.cpp:16355`). Minimal patch about 20 lines (`patches/vulkan-android-1.1-mali.diff`); answers then match the CPU. A draft upstream issue is in `docs/draft-upstream-issue.md` (not published).
- **VERDICT: the GPU is slower than the CPU in every variant, cold or warm.** Tight crop TTFT: CPU 1.9-2.4 s; Vulkan LM-only 9.3 s (warm 7.5 s); Vulkan all 14.2 s (warm 13.3 s). One-tile frame: CPU 6.6-8.0 s; Vulkan 17.6 s LM-only, 31.2 s all.
- **VERDICT: cause is raw compute: about 7-11 GOPS in `ggml-vulkan` on this 2-core Mali** (LM prefill about 8 GOPS vs 103-111 on the CPU, 12x slower; encoder 7-11 vs 37-40, 4-5x slower; generation about 110 ms/token vs 36-80 ms). A timestamped profile shows the CPU waiting on the GPU fence.
- **VERDICT: shader compilation is a separate cold-start cost, not the cause of slowness.** No persistent pipeline cache: 18-33 pipelines, 7-10 s per process, three fused rms-norm kernels about 2 s each (`rms_norm_mul_f32_512`, `rms_norm_mul_rope_f32_f32_128`, `rms_norm_mul_rope_f32_f16_128`). Server ready after 10.7-11.3 s vs 2.1 s on the CPU. `GGML_VK_DISABLE_FUSION=1` halves compile time (10.0 -> 4.7 s) but does not change TTFT.
- **RETRACTED: my earlier claim "warm LM image prefill on the GPU takes 27-39 ms, 12-15x faster than the CPU" was a measurement artefact** (the image decode returns after submitting work; the wait lands on the next text decode).
- **VERDICT: a CPU-encoder + GPU-LM split does not help** (LM is 12x slower on the GPU).
- **VERDICT (OpenCL): not usable on Mali.** The backend accepts only Adreno and Intel; with an experimental allowlist change it still fails to compile kernels (they assume Adreno subgroup features; this driver reports no subgroup shuffle).

## 8. Not done / open
- KleidiAI (default OFF; would need the v1.32.0 archive download; whether dotprod-only kernels are used was not checked), LM quant sweep (Q4_0, Q4_K_M), mmproj F16 vs Q8_0 (mmproj Q4_0 is not possible via the convert script), per-factor sweep from the experiment-2 plan.
- Accuracy of the repacked encoder on all conditions; repack with other mmproj quants.
- `git bisect` (not needed: no regression).
- Other phones (no device farm with a comparable budget SoC was identified).

## 9. Data and method caveats
- N = 16 tags (30 planned; stopped early), one run per tag and condition (round-robin), `-t 4`; the regression, pinning, GPU and repack tests use one image per condition with 5-10 repeats.
- Source images are warped shelf displays from a private customer dataset; only numbers are published, not photos. B and A2 are host-pre-resized frames (about 70 and about 250 tokens); the default tiled path is condition A.
- Achieved GOPS use a simple FLOP formula (short-conv layers approximated as dense, patch embedding and projector ignored) and are indicative. The int8 dotprod peak of about 280 GOPS for 2 A76 cores is an estimate, not a measurement.
- `llama-server` numbers are 10-20% above the CLI numbers (HTTP, base64, hot CPU): compare within a table.
- Answer normalisation: `$`, trailing `.00`/`,00` removed; a separator before exactly three digits is a thousands separator.

## 10. Publishing state
Nothing was pushed or published. Items waiting for an explicit decision: the repo commit/push (branch name, whether to keep the real prices in the CSV), the upstream issue draft, and the three local patches (`vulkan-android-1.1-mali.diff`, `mtmd-encoder-repack-experiment.diff`).
