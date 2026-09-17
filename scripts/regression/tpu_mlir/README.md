# tpu-mlir real-model regression

Runs onnxsim over real YOLOX detector exports and pushes the simplified graph
all the way through [tpu-mlir](https://github.com/sophgo/tpu-mlir)'s real
ONNX ingestion, `tpuc-opt` canonicalization, and `pymlir` interpretation --
the same real toolchain the Milk-V Duo (CV1800B) and other Sophgo boards use
to compile a `kmodel`/`cvimodel`.

See [`tests/test_tpu_mlir_integration.py`](../../../tests/test_tpu_mlir_integration.py)
for the synthetic-graph side of this (the small, CI-gated regression for
tpu-mlir's own `OnnxConverter.model_simplify()` -- the exact code path that
calls `onnxsim.simplify()`, with code comments in tpu-mlir's own source citing
real bugs it hit running "yolox" and "ppyolo_tiny" against the old onnxsim
version it pins). This script is the real-model counterpart: same YOLOX
checkpoints already confirmed clean against onnxsim in
[`scripts/regression/yolox/`](../yolox/), pushed one stage further into a real
downstream compiler instead of stopping at `onnxsim.simplify()`'s own check.

## Why YOLOX and not ppyolo_tiny

tpu-mlir's own code comments name two bugs: one against "yolox", one against
"ppyolo_tiny". YOLOX is covered here. `ppyolo_tiny` is not: a real
PaddleDetection `ppyolo_tiny` checkpoint was obtained and traced far enough to
confirm its detection head is NMS-based -- the same category of graph as
[onnxsim issue #60](https://github.com/onnxsim/onnxsim/issues/60) (a
`PrepareForReduce` crash from onnxsim's random-input constant folding
degenerating an NMS output to a zero-sized dimension) -- but paddle2onnx has
never supported the specific op version (bare `multiclass_nms`, predating
`multiclass_nms3`) that particular 2021 checkpoint uses, on any released
paddle2onnx version (0.3.1 through the current 2.1.0). That's a permanent
paddle2onnx limitation the checkpoint itself runs into, not something a
regression harness can route around -- see this project's own investigation
notes for the full trace (a legacy-format `paddle2onnx export` first needs an
old-enough paddle2onnx to read the pre-PIR `__model__`/`__params__` binary
format at all, then a new-enough one to know `multiclass_nms`; no released
version is both).

## Requirements

`tpu_mlir` is heavy (~310MB wheel), Python-3.10-only, and **only really runs
on Ubuntu 22.04** -- its compiled `tpuc-opt`/`pymlir` native extensions are
linked against Ubuntu 22.04's glibc/libstdc++ and segfault against a newer
host's ABI (confirmed directly against Ubuntu 24.04, with or without
`LD_LIBRARY_PATH` tricks to mix vendored and host libraries -- see
`.github/workflows/backend-integration.yml`'s `tpu_mlir` job, which runs
inside a pinned `ubuntu:22.04` container for exactly this reason). Run this
script the same way: inside an `ubuntu:22.04` container/chroot, not directly
on a newer host.

```bash
# inside ubuntu:22.04
apt-get update && apt-get install -y python3 python3-pip
pip install "tpu_mlir[onnx]"    # pins onnx==1.14.1, onnxruntime==1.16.3,
                                # onnxsim==0.4.17, numpy==1.24.3, protobuf==3.20.3
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install loguru thop tabulate tqdm psutil opencv-python-headless
git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git

# swap in the onnxsim under test (current onnxsim, not tpu_mlir's pinned 0.4.17)
pip install --force-reinstall --no-deps <onnxsim-under-test>.whl
```

## Running

```bash
PYTHONPATH=$PWD/YOLOX python scripts/regression/tpu_mlir/run_tpu_mlir_regression.py \
    --download --weights-dir yolox-weights --workdir tpu-mlir-reg-work \
    --opset 13 --output tpu-mlir-regression.csv
```

Exit code is non-zero if onnxsim's own check fails on any variant, or if
tpu-mlir's ingestion/canonicalization/interpretation of the *simplified*
graph raises or disagrees numerically with onnx's own reference evaluator on
the original graph.

## Current status

Not yet run end-to-end in this repo's own environment: this sandbox's host OS
(Ubuntu 24.04) hits exactly the `tpuc-opt` ABI wall described above, so no
`tpu-mlir-regression.csv`/results table is checked in here yet. The
`tests/test_tpu_mlir_integration.py` synthetic-graph tests *were* run to
completion up to that same wall (confirmed: tpu-mlir's `OnnxConverter`
construction, `onnxsim.simplify()` call, and raw Top MLIR generation all
succeed cleanly against current onnxsim; only the native `tpuc-opt`
canonicalization step is blocked by the OS mismatch) -- see that module's
docstring. A recorded run of this script needs the `ubuntu:22.04` container
described above, e.g. via the CI job.
