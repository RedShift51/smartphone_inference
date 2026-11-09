# smartphone_inference

Some notes and scripts from a small set of measurements I did out of curiosity: where does the time go when a small vision-language model (LFM2-VL-450M, via llama.cpp) reads a price from a photo on an inexpensive phone (Redmi 13, Helio G99, CPU)?

Please treat the numbers as rough. It is 16 test images, one phone, one model, mostly one run per image and condition, so I would not trust the small differences, and I may well have made mistakes. If you spot one, I would be glad to hear about it.

## What I saw (in my runs)

- Most of the time before the first token went to the vision encoder: roughly 70-78%, depending on the image size.
- With the same token budget, a tight crop around the price tag read the price in 10 of 16 images, against 4 of 16 for the whole shelf shrunk to the same size, at about the same latency. That is a small sample (p about 0.07), so it is a hint, not a conclusion.
- I first suspected a llama.cpp regression, but an older build (tag b10516) and the current one were about equally fast here.
- The encoder's weights seem not to go through the repacked int8 kernels that the language model uses, which would explain why it is slower per operation. A small local experiment that places them there made the encoder about 1.6-1.8 times faster with the same answers on the images I checked. It is a quick experiment, tested with only this one model; it has not been reviewed or submitted to llama.cpp.
- On this phone's GPU (Mali-G57) the Vulkan backend crashed at first; after a small local fix it ran, but it was slower than the CPU, and OpenCL did not work with this GPU. This is specific to this GPU and driver, so it says little about other phones.

The details, caveats and a few things I got wrong along the way (and corrected) are in `results/VERDICTS.md`.

## What is here

- `results/`: summary (`VERDICTS.md`, `summary.md`, `final_report.md`), raw measurements (`*.csv`), and the charts (`results/post/`).
- `docs/`: plans, deviations from the plans, notes from reading the llama.cpp code, and notes on the Vulkan/OpenCL attempts. `docs/draft-upstream-issue.md` is only a fact sheet for myself, not a finished report.
- `patches/`: the small local llama.cpp changes mentioned above, as diffs (they are experiments, not polished patches).
- `scripts/`: the scripts I used to run and analyse the measurements.

## Reproducing (roughly)

1. Clone llama.cpp (`https://github.com/ggml-org/llama.cpp`; I used commit `a691b2345844ce1b430201fc5fd404ce062e05f2`) and build `llama-mtmd-cli` for arm64 with the Android NDK:
   `cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16" -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16" -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF && cmake --build build-android -j --target llama-mtmd-cli`
2. Download `LiquidAI/LFM2-VL-450M-GGUF` (the Q8_0 model and its Q8_0 mmproj) and push the binary and models to `/data/local/tmp/vlm` with adb.
3. `pip install -r requirements.txt`; set `ADB=/path/to/adb` if adb is not on your PATH.
4. The scripts in `scripts/` expect the phone to be connected; `vlm.py` is the small adb helper the others use.

The test photos are not included (they are not mine to share), so the image-based runs cannot be repeated from this repository alone; the numbers and scripts are here for reference.

## Notes

- Experiments were done on a warm phone with battery and CPU temperature logged; absolute times drift by 10-20% with temperature, so I only compared runs from the same session.
- Parts of the analysis and scripts were written with the help of an AI assistant (Claude). I checked the numbers against the raw data, but some mistakes may remain.
- Thanks to the llama.cpp and Liquid AI teams for the code and the model.
