# smartphone_inference

Measuring where time goes (visual encoder / image prefill / text prefill) for the LFM2-VL-450M VLM on a budget phone
(Redmi 13, Helio G99, CPU only) with `llama-mtmd-cli` from llama.cpp.

- **All verdicts: `results/VERDICTS.md`.**
- Experiment 1: crop vs full frame (conditions A, B, A2, C, D, E). Plan in docs, results in `results/`.
- Experiment 2: compute-bound breakdown (code reading done, profiling and config sweep pending). Plan: `docs/experiment2-compute-bound-breakdown.md`.

## Setup
1. `git clone https://github.com/ggml-org/llama.cpp` (results use commit `a691b2345844ce1b430201fc5fd404ce062e05f2`) and build for arm64:
   `cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16" -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16" -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF && cmake --build build-android -j --target llama-mtmd-cli`
2. Models: `LiquidAI/LFM2-VL-450M-GGUF` (Q8_0 LM + mmproj Q8_0) into `models/`; push binary and models to `/data/local/tmp/vlm` with adb.
3. `pip install -r requirements.txt`; set `ADB=/path/to/adb` if adb is not on PATH.

## Scripts (`scripts/`)
- `fetch_r2.py <prefix> <out>`: read-only download from Cloudflare R2; credentials are read from the macOS Keychain (service name from `R2_KEYCHAIN_SERVICE`, default `r2-credentials`; keys `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`), nothing is stored in the repo.
- `prep_exp1_data.py`: builds the 30 test tags (full / full_b / full_a2 / tight / ctx4) and `gt.csv` from `data/price_tag_dataset`.
- `vlm.py`: adb runner and log parser. `calib.py`, `thread_calib.py`: calibration. `exp1_main.py`: main round-robin run (resumable). `analyze_exp1.py`: summary table and plots.

## Data
Photos are customer data and are not in the repo (`data/` is ignored). `results/` holds only numbers and plots.
Stopped early: N=16 fully measured images of the planned 30; see docs for deviations.
