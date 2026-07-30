# YOLOX regression

Runs `onnxsim` over [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
detector exports and checks that the simplifier does not crash, hang, or fail
its own correctness check.

YOLOX is a canonical onnxsim consumer: its `tools/export_onnx.py` calls
`onnxsim.simplify(...)` and asserts the check passes. This harness reproduces
that path across the nano / tiny / s variants at a given opset, and records the
node-count reduction, validity, wall-clock, and peak RSS — the same
"what counts as a failure" definition as the onnxmodelzoo harness one level up
in [`scripts/regression/`](../README.md).

Unlike that harness (which downloads ready-made ONNX from Hugging Face), YOLOX
ships as PyTorch, so this one needs torch + the YOLOX source to produce the
graph before handing it to onnxsim.

## Running

```bash
# onnxsim under test + its runtime deps
pip install onnx onnxruntime .          # or an onnxsim wheel

# export toolchain (not onnxsim deps)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install loguru thop tabulate tqdm psutil opencv-python-headless
git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git

# --download fetches the official 0.1.1rc0 checkpoints on first run
PYTHONPATH=$PWD/YOLOX python scripts/regression/yolox/run_yolox_regression.py \
    --download --weights-dir yolox-weights --workdir yolox-reg-work \
    --opset 11 --output yolox-regression-opset11.csv
```

Exit code is non-zero if any variant crashes, times out, or fails onnxsim's
correctness check. Re-run with `--opset 17` (or any supported opset) to cover a
different export path; onnxsim's constant-folding / optimizer behaviour varies
by opset.

See [`RESULTS.md`](./RESULTS.md) for a recorded run.
