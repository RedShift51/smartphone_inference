# Experiment 2: preliminary observations (smoke test, not measurements)

Device: Xiaomi 23117RA68G (Redmi 13), SoC MT6789 (Helio G99), Android 13, 11.4 GB RAM, arm64-v8a.
Cores (`cpuinfo_max_freq`): cpu0-5 at 2.0 GHz, cpu6-7 at 2.2 GHz, so the big cores are cpu6,7 (mask `0xC0`).
llama.cpp `a691b2345844ce1b430201fc5fd404ce062e05f2`, CPU-only build (`--list-devices` is empty), KleidiAI off.
Model: LFM2-VL-450M Q8_0 + mmproj Q8_0 (Hugging Face `LiquidAI/LFM2-VL-450M-GGUF`); Q4_0, F16 and mmproj F16 are also downloaded.
mmproj hparams: image_size 256, min_pixels 65536 (64 tokens), max_pixels 262144 (256 tokens).
Flags available: `--image-min-tokens`, `--image-max-tokens`, `-dev`, `--list-devices`, `--mmproj-offload`, `-C` / `-Cr`.

Smoke run (`test-1.jpeg`, 576x416 -> 234 tokens, `-t 6`, one run, no temperature control):
encode about 4.8 s, image decode in the LM 1029 ms, prompt eval 6018 ms for 251 tokens, eval 36 ms/token.
The encoder is about 4.7x more expensive than the LM prefill of the same tokens.

simpleperf (call graph on, not yet split into clip/llama): `tinyBLAS_Q0_ARM<q8_0>::gemm` 30.0%, `ggml_graph_compute_thread` 24.2%, `flash_attn_ext` 12.3%, `ggml_gemm_q8_0_4x4_q8_0` 10.9%.
Hypothesis (unverified): tinyBLAS is the encoder (no repack), the 4x4 gemm is the LM (repacked). Needs attribution through the call graph.

GPU: the device has `libOpenCL.so` and `vulkan.mali.so`; the current build has no GPU backend.
