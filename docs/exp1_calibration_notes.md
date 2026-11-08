# Experiment 1: calibration notes

Setup: LFM2-VL-450M Q8_0 (LM + mmproj), llama.cpp CPU, 30 candidate tags from a private dataset (not in repo), seed 42.

Deviations from the plan:
1. Source images are display warps (long side <= 1600), not original photos.
2. Full frames are tiled by the model: a 544x1024 frame gives 2 tiles of 512x512 plus an overview of 352x672, i.e. 256+256+231 = 743 image tokens and 3 encoder passes.
3. `--image-max-tokens 64` only shrinks the overview (to 50 tokens); tiles stay. So condition B uses a frame pre-resized on the host to about 64 tokens, and A2 a frame pre-resized to one tile (<= 256 tokens).
4. Small crops are not upscaled to exactly 64 tokens (aspect ratio is kept): tight gives about 72-78 tokens; min=max=16 gives 20-21.
5. `thermal_zone*` temperatures are not readable without root; only battery temperature was used.
6. Answer normalisation: a separator followed by exactly three digits is a thousands separator (`$1,210` = 1210).
7. The run was stopped early: 16 fully measured images instead of 30.

Thread calibration (condition A, 3 rounds, interleaved): median prompt eval 21.5 s (`-t 2`), 21.0 s (`-t 4`), 21.5 s (`-t 6`). The differences are smaller than thermal drift (battery 37.3 -> 38.9 C). The main run used `-t 4`.
