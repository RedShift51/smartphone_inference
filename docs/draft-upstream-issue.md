> **DO NOT PASTE AS-IS.** llama.cpp's CONTRIBUTING.md prohibits using AI to write issues, PR descriptions or replies, and requires disclosing AI use. This file is a *fact sheet* (symptoms, root-cause locations, measurements) to be rewritten in your own words before anything is posted.

# DRAFT (not published): ggml-vulkan crashes on Vulkan 1.1 drivers (Mali-G57)

Title: `Vulkan: segfault on devices whose driver reports Vulkan 1.1 (Mali-G57, Android 13)`

llama.cpp `a691b2345844ce1b430201fc5fd404ce062e05f2`, Android arm64 (NDK 28.2, cross-compiled, `-DGGML_VULKAN=ON`), Redmi 13 (Helio G99), Mali-G57 MC2. `--list-devices` shows `Vulkan0: Mali-G57 MC2`; any model run segfaults while the model is loading (also with `-ngl 0`).

The driver reports `apiVersion` 1.1.177 for the physical device (`cmd gpu vkjson`), with `VK_KHR_buffer_device_address` and `VK_KHR_timeline_semaphore` only as extensions. `ggml-vulkan` only requires the *instance* version >= 1.2 and then calls core 1.2 entry points through a `vulkan.hpp` dispatcher initialised from the instance, so on a 1.1 driver those names resolve to loader trampolines with a NULL driver entry (crash at `pc = 0`).
Three crashes, in order (tombstones symbolised):
1. `ggml_vk_create_buffer` -> `getBufferAddress` (`ggml-vulkan-buffers.cpp:202`): `vkGetBufferDeviceAddress` is NULL although `vk12_features.bufferDeviceAddress` is true.
2. `ggml_vk_submit` -> `vkQueueSubmit` (system `libvulkan.so`) with a queue from `vkGetDeviceQueue2` (`ggml-vulkan.cpp:1050`); `vkGetDeviceQueue` works.
3. `ggml_backend_vk_device_event_synchronize` -> `waitSemaphores` (`ggml-vulkan.cpp:16355`): `vkWaitSemaphores`, `vkSignalSemaphore`, `vkGetSemaphoreCounterValue` are NULL.
Also, `VkPhysicalDeviceVulkan11/12Features` are declared without initialisation (only `sType`/`pNext` set) and read back when the driver does not fill them.

A local patch (about 20 lines: gate BDA on device `apiVersion >= 1.2`, enable `VK_KHR_timeline_semaphore` and route the three timeline calls to their KHR entry points on < 1.2, use `vkGetDeviceQueue`) makes the backend run correctly; results match the CPU. Suggested upstream behaviour: either these fallbacks, or reject devices below Vulkan 1.2 with a clear message instead of crashing.
Note: with the patch the backend works but is slower than the CPU on this small GPU (about 8 GOPS effective on LFM2-VL prefill vs about 105 on the CPU), and every process spends several seconds compiling shaders because there is no persistent pipeline cache (three fused `rms_norm_mul[_rope]` pipelines take about 2 s each on this driver).
