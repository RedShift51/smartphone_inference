# Vulkan and OpenCL on Mali-G57 (Helio G99): root causes and measurements

Device: Redmi 13 (Helio G99), Mali-G57 MC2. llama.cpp `a691b2345844ce1b430201fc5fd404ce062e05f2`. Local experiments only; nothing was published.

## Vulkan segfault: three independent causes
The Mali driver reports **Vulkan 1.1.177** for the physical device (`cmd gpu vkjson`), with `VK_KHR_buffer_device_address` and `VK_KHR_timeline_semaphore` available only as extensions. `ggml-vulkan` only checks that the *instance* version is >= 1.2 (`ggml-vulkan.cpp`, `api_version < VK_API_VERSION_1_2`) and then uses 1.2 core entry points. The `vulkan.hpp` dispatcher is initialised from the instance only (`VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance)`), so core 1.2 names resolve to loader trampolines whose driver entry is NULL on a 1.1 driver (crash with `pc = 0`).

Crash chain found with tombstones symbolised by `llvm-symbolizer`:
1. `ggml_vk_create_buffer` -> `vk::Device::getBufferAddress` (`ggml-vulkan-buffers.cpp:202`): `device->buffer_device_address` is taken from `vk12_features.bufferDeviceAddress`, but `vkGetBufferDeviceAddress` (core 1.2) is NULL.
2. `ggml_vk_submit` -> `vkQueueSubmit` inside the system `libvulkan.so` (`ggml_backend_vk_event_record`): the queue obtained with `vkGetDeviceQueue2` (`ggml-vulkan.cpp:1050`) is unusable with the Android loader; `vkGetDeviceQueue` works.
3. `ggml_backend_vk_device_event_synchronize` -> `vk::Device::waitSemaphores` (`ggml-vulkan.cpp:16355`): `vkWaitSemaphores` / `vkSignalSemaphore` / `vkGetSemaphoreCounterValue` (core 1.2 timeline semaphores) are NULL.

Minimal local patch that makes it run (`patches/vulkan-android-1.1-mali.diff`, 20 insertions):
- `buffer_device_address = vk12_features.bufferDeviceAddress && device->properties.apiVersion >= VK_API_VERSION_1_2`.
- For `apiVersion < 1.2`: enable `VK_KHR_timeline_semaphore` and, after `createDevice`, set `vkWaitSemaphores`, `vkSignalSemaphore`, `vkGetSemaphoreCounterValue` of the default dispatcher to their `KHR` counterparts.
- Use `vkGetDeviceQueue` instead of `vkGetDeviceQueue2` when internally synchronized queues are not used.
Not needed: zero-initialising `VkPhysicalDeviceVulkan11/12Features` (still good hygiene, they are read uninitialised when the driver does not fill them).
The proper upstream behaviour would be either these fallbacks or rejecting devices below Vulkan 1.2 with a clear error instead of a segfault.

## Vulkan results with the patch (D image 09 and A2 image 10, 3 repeats, median)
Answers are identical to CPU on every run (D 1290, A2 1180).

| cond | config | encoder ms | image prefill ms | text prefill + shader compile ms | TTFT ms |
|---|---|---|---|---|---|
| D | CPU `-t 4` | 1686 | 412 | 200 | 2296 |
| D | CPU 2 big cores | 1258 | 423 | 204 | 1885 |
| D | Vulkan (all) | 7321 | 653 | 6373 | 14245 |
| D | Vulkan LM only, encoder CPU | 1570 | 1073 | 6445 | 9346 |
| A2 | CPU `-t 4` | 5055 | 1298 | 252 | 6605 |
| A2 | Vulkan (all) | 19227 | 670 | 12662 | 32559 |
| A2 | Vulkan LM only, encoder CPU | 5027 | 1136 | 11521 | 17634 |

## Why it is slow: the Mali GPU delivers about 7-11 GOPS in ggml-vulkan (CPU: 37-110)
Earlier I reported "warm LM image prefill 27-39 ms, 12-15x faster than the CPU". **That was a measurement artefact.** `llama_decode` of an image batch (no logits) returns once the work is submitted, without waiting for the GPU; the wait is paid by the next text decode (which needs logits). A timestamped `simpleperf` sample of the window after the image decode (6.05 s of wall time) shows only 2.3 s of CPU samples, with the main thread inside `ggml_vk_wait_for_fence`: the CPU is waiting for the GPU, which really computes for about 6 s.
Effective throughput from the same runs (FLOP from the formulas in `results/summary.md`):
- LM prefill on Vulkan (about 100 tokens, 2 x 287M x tokens, about 57 GOP in about 7 s): about 8 GOPS. CPU: 103-111 GOPS. The log's own number agrees: 97 ms per prompt token on the GPU vs about 5.5 ms on the CPU.
- Vision encoder on Vulkan: D 52 GOP in 7.3 s = 7 GOPS, A2 207 GOP in 19.2 s = 11 GOPS. CPU: 37-40 GOPS.
- Token generation: about 110 ms per token on the GPU (CPU 36-80 ms).
So this is plain compute throughput of a 2-core Mali-G57 through the generic (no cooperative-matrix, no int8-dot) shaders, not a driver bug.

Second, separate cost: shader compilation. Pipelines are compiled lazily and there is no persistent pipeline cache (no `VkPipelineCache` in `ggml-vulkan`). Instrumented run (`GGML_VK_COMPILE_LOG`, local print in `vkCreateComputePipeline`): 18-33 pipelines per process, 7-10 s in total, of which three fused rms-norm kernels take about 2 s each: `rms_norm_mul_f32_512` (1.9 s), `rms_norm_mul_rope_f32_f32_128` (2.3 s), `rms_norm_mul_rope_f32_f16_128` (2.3 s). Most of this happens in the model-load warm-up (`load time` about 9 s instead of about 2.5 s on the CPU), a little (about 1 s: matmul and flash-attention variants) inside the first image decode. `GGML_VK_DISABLE_FUSION=1` halves the compile total (10.0 s -> 4.7 s; the slowest pipelines drop to about 0.6 s) but TTFT does not change, because TTFT is dominated by GPU compute.

Reading: on this phone the GPU is about 4-5x slower than the CPU for the encoder and about 12x slower for the LM prefill; a CPU-encoder + GPU-LM split would not help either.

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

## OpenCL
The OpenCL backend only accepts Adreno and Intel GPUs (`ggml_opencl_is_device_supported`, `ggml-opencl.cpp` ~8941-8954); on Mali it prints `unsupported GPU` and offers no device. With an experimental allowlist change it starts but kernel compilation fails (`Compiler frontend failed`, undefined `N_SIMDGROUP`, `N_DST`, `N_SIMDWIDTH`): the kernels are loaded unconditionally and assume Adreno subgroup width and shuffle support; this driver reports no subgroup shuffle. Porting would be real kernel work, not a flag.
