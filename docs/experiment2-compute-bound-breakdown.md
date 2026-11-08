# Experiment 2: where encoder and prefill time goes on a mobile CPU

Continuation of experiment 1. Same build, phone, LFM2-VL-450M model and measurement protocol (airplane mode, temperature control, interleaving).
Same rule: **verify in code and logs, do not guess**; record any discrepancy.

## Question

On a budget ARM phone (2x Cortex-A76 + 6x Cortex-A55, dotprod, no i8mm) the visual encoder and prefill are compute-bound. Instructions set the ceiling: int8 via `SDOT` gives roughly 4x more operations per cycle than fp32 NEON on the same core.

Hypotheses:
- **G1.** The LM goes through the fast path (Q4_0 -> q4_0_4x4 repack, KleidiAI) while the visual encoder (`clip.cpp`) does not, so the most expensive phase runs furthest from peak.
- **G2.** Weight format sets prefill speed through kernel availability, not bit count: repacked Q4_0 may beat Q8_0 and Q4_K_M on prefill.
- **G3.** The slow A55 cores hold back compute-bound phases: two explicitly pinned big cores may match eight threads on prefill and encoder, while decode prefers another configuration.

## Step 1. Code reading (before measuring)

In the same llama.cpp commit, find and record with `file:line`:
1. How many threads the encoder uses in `tools/mtmd`: `-t`, `-tb` or its own.
2. Whether the encoder uses extra CPU buffer types (repack, KleidiAI), compared with the LM.
3. Which mmproj quantisation types are supported and how to quantise it.
4. Which types `ggml/src/ggml-cpu/repack.cpp` repacks with NEON + dotprod and no i8mm.
5. The KleidiAI build option: exact name, Android default, accelerated types.

Record conclusions briefly. If the code shows G1 is wrong, say so immediately; step 4 would then be pointless.

## Step 2. Core topology

```bash
adb shell 'for c in /sys/devices/system/cpu/cpu[0-9]*; do echo $c $(cat $c/cpufreq/cpuinfo_max_freq); done'
```

Identify the big cores (expected: two with the highest frequency). Pin with `taskset` on the device or `--cpu-mask` / `--cpu-range` (check `--help`).

## Step 3. Profile of one run

Input: the full frame from experiment 1 (condition A, most tokens). Profile with `simpleperf`:

```bash
adb shell 'cd /data/local/tmp/vlm && simpleperf record -o perf.data --call-graph fp \
  ./llama-mtmd-cli -m <LM> --mmproj <MMPROJ> --image <A.jpg> -p "<prompt>" --temp 0 -n 16 -t <best>'
adb shell 'cd /data/local/tmp/vlm && simpleperf report -i perf.data --sort symbol' | head -60
```

If `--call-graph fp` fails, use `dwarf` or no graph. Need the **top 20 functions by time**, each labelled encoder (`clip`) or LM (`llama`). Main question: which matmul kernels are hot in the encoder (plain F16/Q8 vec_dot, repacked gemm, KleidiAI).

## Step 4. Parameter sweep

One input (condition A), 5 repeats per configuration, configurations interleaved, temperature control as in experiment 1. **One factor at a time from the baseline**, then combine the best.

Baseline: LM Q8_0, mmproj Q8_0, best thread count from experiment 1.

| Factor | Values |
|---|---|
| LM quant | Q4_0, Q8_0, Q4_K_M (from the repo, or `llama-quantize` from F16) |
| mmproj quant | F16, Q8_0, and Q4_0 if supported (see step 1) |
| KleidiAI | build flag off / on (LM Q4_0 only) |
| Threads | 2 pinned to big cores; 4; 8 unpinned. Vary `-t` and `-tb` separately if the encoder uses one of them |
| Mobile GPU | Vulkan (or OpenCL) build; `-dev` and `--mmproj-offload` for LM and encoder separately (separate item, not in the baseline) |

If the encoder does not get the fast path and time allows, check whether it can be enabled without code changes (flag, buffer type). **Do not modify llama.cpp** without agreement.

## Step 5. Metrics

As in experiment 1 (encoder time, image prefill, text prefill, decode, tokens), plus **achieved performance**:
- Encoder: FLOP ~ L * (24*N*d^2 + 4*N^2*d), with L layers, width d, N patches **before** pixel shuffle (from mmproj metadata or the log). Divide by encoder time for GFLOPS.
- LM prefill: FLOP ~ 2 * (LM params without embeddings) * (tokens), giving GOPS.

Compare with the rough peak of 2x A76 at about 2.2 GHz: fp32 NEON ~70 GFLOPS, int8 dotprod ~280 GOPS (estimates, not measurements). Report percent of peak per phase.

Quality check: for the best configuration, run the experiment 1 set and compare accuracy with the baseline. Record any drop of more than 2 tags out of 30.

## Deliverables

1. Step 1 conclusions with `file:line`.
2. simpleperf top 20 labelled encoder / LM.
3. Configuration table: median and spread per phase, GFLOPS and percent of peak.
4. Summary: baseline TTFT vs best combination, and how much each factor contributed.
5. Anything odd: crashes, answer mismatches between quants, unexpected hot functions.
