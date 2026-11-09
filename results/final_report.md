# Where does a small VLM spend its time on a budget phone? (post notes)

Setup: Redmi 13 (Helio G99, 2x A76 + 6x A55, Mali-G57 MC2), LFM2-VL-450M Q8_0 (LM and mmproj), llama.cpp `llama-mtmd-cli`, CPU, task = read a 4-digit price from a price tag. N = 16 tags (one fully measured run per tag and condition, round-robin, battery and CPU temperature recorded before each run, `-t 4`).

## Headline numbers
1. **The vision encoder is 70-78% of time-to-first-token.** Median TTFT for a tight tag crop is 2.0 s: encoder 1.4 s, image tokens in the LM 0.4 s, prompt text 0.2 s. Encoder cost is about **20 ms per image token** (R^2 = 0.99); the LM prefill is about 5.5 ms per token.
2. **A crop reads the price better than a squeezed full frame at the same cost.** Same token budget (72 vs 74 tokens) and the same latency (about 2.0 s), but the exact price was read in **10/16** cases with a tight tag crop vs **4/16** with the whole shelf downscaled (paired sign test: 7 vs 1, p = 0.07; suggestive at this N). A crop with context: 8/16.
3. **Full frames get expensive fast.** A full shelf frame that fits one tile (about 250 tokens) takes 6.6 s. The default tiling splits larger frames into several 512x512 tiles plus an overview: median 6.5 s, tail up to 49 s (1776 tokens). `--image-max-tokens` does not limit tiling.
4. **Going lower costs accuracy.** A tight crop at 20 tokens is 0.8 s TTFT but exact reads drop to 5/16 (9/16 if only the digit sequence must match; that metric is more lenient and allows false matches).
5. **Honest TTFT with the detector:** adding the amortised price-tag detector (805 ms per frame on CPU, about 10 tags per frame, about 78 ms per tag) gives about 2.1 s for the tight crop.

## Why the encoder is slow: architecture, not a regression
- Current llama.cpp (2026-11-08) vs tag b10516 (2026-08-19): encoder time ratio 0.97-0.99 (5 alternating repeats, same answers). A reported 2.7x regression (issue #27649, closed as not planned) does not show up on this phone and model.
- The encoder runs a generic int8 kernel (`tinyBLAS_Q0_ARM`, 37-40 GOPS achieved), the LM runs the repacked `q8_0_4x4` kernel (103-111 GOPS). Profiling shows the two pools of threads run disjoint kernels. Reason in the code: the mmproj weights are placed in the default CPU buffer type (no repack), the LM weights in the extra buffer types.
- Thread count: 2 threads pinned to the two big cores (`taskset c0`) cut the encoder from 1485 to 1275 ms (-14%) and TTFT by about 10% versus `-t 4` unpinned, with a much tighter spread. `-t 8` is not better. Unpinned threads spent only about 37% of their running time on the big cores.

## Encoder fix (local experiment)
- Putting the vision-encoder weights into the same CPU repack buffer type the LM already uses makes the encoder 1.6-1.8x faster and TTFT 1.4-1.45x faster with identical answers (tight crop: 2.39 s -> 1.66 s, one-tile frame: 7.9 s -> 5.6 s; 10 interleaved repeats, paired 95% CIs in `VERDICTS.md`). Experimental patch, not upstreamed.

## Mobile GPU (Mali-G57)
- Vulkan crashed with a segfault. Root causes (all because the Mali driver is Vulkan 1.1.177 while `ggml-vulkan` assumes 1.2 core functions): buffer device address, timeline semaphores and a queue obtained with `vkGetDeviceQueue2`. A 20-line local patch makes it run, with answers identical to the CPU.
- It is not faster: it is 5-12x slower. One-shot TTFT on the tight crop: CPU 1.9-2.3 s, Vulkan LM-only 9.3 s, Vulkan all 14.2 s. The Mali-G57 delivers only about 7-11 GOPS in `ggml-vulkan` (LM prefill about 8 GOPS vs 103-111 on the CPU, encoder 7-11 vs 37-40). A profile shows the CPU waiting on the GPU fence while the GPU computes.
- On top of that every process compiles its shaders from scratch (7-10 s, no pipeline cache; three fused rms-norm kernels take about 2 s each), mostly during model load.
- Warm does not rescue it: in a long-lived server, after warm-up, the tight crop still takes 7.5 s (LM on GPU) and 13.3 s (all on GPU) vs 2.3-2.5 s on the CPU. Cold start adds about 9 s (server ready after 10.7-11.3 s vs 2.1 s on the CPU).
- OpenCL: the backend only supports Adreno and Intel; an experimental change to accept Mali still fails to compile the kernels.

## Caveats (say these in the post)
- N = 16 tags, one measured run per tag and condition (warm-up runs excluded); the regression and pinning tests use one image per condition and 5 repeats.
- Test photos are warped shelf displays from a customer dataset: only numbers are published.
- Achieved GOPS use a simple formula (short-conv layers approximated as dense) and are indicative.
- Full-frame conditions use frames pre-resized on the host to about 70 or about 250 tokens; the model's default tiling is reported separately.

## Files
Plots (`results/post/`): `time_by_phase.png` (main), `encoder_vs_tokens.png`, `accuracy_vs_tokens.png`, `full_shelf_tiled_ttft.png`. Tables and raw data: `summary.md`, `exp1_runs.csv`, `exp3_regression.csv`, `exp3_pinning.csv`, `exp3_gpu.csv`. Vulkan: `docs/vulkan-mali-findings.md`, `patches/vulkan-android-1.1-mali.diff`, `docs/draft-upstream-issue.md`.
