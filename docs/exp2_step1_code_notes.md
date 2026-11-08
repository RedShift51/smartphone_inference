# Experiment 2, step 1: code reading notes

llama.cpp commit `a691b2345844ce1b430201fc5fd404ce062e05f2` (shallow clone). Static reading only; nothing here was verified on the device.

1. **Encoder threads.** The encoder uses `-t`, not `-tb`: `tools/mtmd/mtmd-cli.cpp:161` (`mparams.n_threads = params.cpuparams.n_threads`), then `mtmd.cpp:652` and `:2238` into `clip_image_batch_encode`. It has its own threadpool, `tools/mtmd/clip.cpp:162-176` (`poll = 0`, no cpumask).
   Consequence for G3: `--cpu-mask` / `--cpu-range` (`common/arg.cpp:1507,1517`) most likely do not pin encoder threads; use `taskset` and verify on the device.
2. **Extra buffer types.** The encoder does not use them: weights are allocated in the backend's default buffer type, `clip.cpp:4815-4816`, and the scheduler also uses default buffer types (`clip.cpp:257-258`). The LM does use them: `src/llama-model.cpp:1262-1275` (`ggml_backend_dev_get_extra_bufts`, disabled by `no_extra_bufts`, `common/common.h:591`). By the code, G1 holds: the encoder has no repack / KleidiAI path, the LM has.
3. **mmproj quantisation.** No path found in `tools/quantize/quantize.cpp` or `tools/mtmd`. What remains is `convert_hf_to_gguf.py --mmproj --outtype {f32,f16,bf16,q8_0,...}`. mmproj Q4_0 is not available, so that configuration is skipped.
4. **Repack with NEON + dotprod, no i8mm** (`ggml/src/ggml-cpu/repack.cpp`, trait selection around lines 4979-5120): Q4_0 -> `q4_0_4x4` (needs `ne[1] % 4 == 0`); Q4_K -> `q4_K_8x4` (`ne[1] % 8 == 0`); IQ4_NL; **Q8_0 -> `q8_0_4x4`** (5110-5120). The 4x8 / 8x8 variants need `matmul_int8` (i8mm) and are not selected on this phone.
   Correction to G2: LM Q8_0 is also repacked on this phone, so "Q4_0 has a fast kernel, Q8_0 does not" is wrong. Differences between quants come from memory bandwidth and kernel cost. Q4_K_M mixes types; which tensors are repacked must be read from the load log.
5. **KleidiAI.** Option `GGML_CPU_KLEIDIAI`, default **OFF** (`ggml/CMakeLists.txt:153`; `docs/build.md:692-698`: enable explicitly for Android arm64-v8a). Supported weight types in `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (~502-506, 692, 821-826): Q4_0, Q4_K, Q8_0, Q6_K, plus F32/F16. At configure time CMake downloads the KleidiAI v1.32.0 archive from GitHub (`ggml/src/ggml-cpu/CMakeLists.txt:653-655`). Whether dotprod-only (no i8mm) kernels are used was not checked.

Conclusion on G1: the code does not refute it. The encoder uses the default buffer without repack; the LM uses extra buffer types. The profile (step 3) will confirm or reject it.
