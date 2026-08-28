# onnx-finetune

A small C++ CLI that fine-tunes an ONNX model in place, using ONNX Runtime's
on-device training API, and exports a normal inference-ready `.onnx` file.
No PyTorch, no re-export from a training framework -- it trains directly
against the ONNX graph.

## How it fits together

Fine-tuning a bare `.onnx` file with ONNX Runtime is a two-step workflow,
split across two different tools because of where each capability lives:

1. **`scripts/generate_artifacts.py`** (Python, one-time, offline) -- builds
   the gradient graph for your model and writes four files: `training_model.onnx`,
   `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint`. This step needs
   Python because gradient-graph construction (`onnxruntime.training.artifacts`)
   is only exposed there -- there is no C/C++ API for it.
2. **`onnx-finetune`** (this C++ binary, runs training) -- loads those four
   files and runs the actual train loop (forward, backward, optimizer step)
   purely through ONNX Runtime's native C++ API, with zero Python dependency.
   It exports the trained result back to a plain `.onnx` file at the end.

This split is deliberate, not a limitation of this tool specifically: it's
how ONNX Runtime's on-device training is designed to be deployed (mobile/edge
apps generate artifacts once during a build step, then ship only the C++/Java/
Swift runtime, never Python).

## Prerequisite: a training-enabled ONNX Runtime build

Neither the `pip install onnxruntime` wheels nor the official prebuilt
release tarballs (the ones under GitHub Releases) include the training C++
API or `onnxruntime.training.artifacts`. You need to build ONNX Runtime from
source with training enabled, once:

```sh
git clone --branch v1.19.2 https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git submodule update --init --recursive

# For generate_artifacts.py (Python bindings + wheel):
python3 tools/ci_build/build.py \
  --build_dir build --config Release --parallel \
  --skip_tests --allow_running_as_root \
  --build_shared_lib --enable_training \
  --enable_pybind --build_wheel

# For onnx-finetune only (no Python bindings needed), drop --enable_pybind
# --build_wheel and --enable_training_apis is enough instead of the fuller
# --enable_training:
python3 tools/ci_build/build.py \
  --build_dir build --config Release --parallel \
  --skip_tests --allow_running_as_root \
  --build_shared_lib --enable_training_apis
```

This is a real ~20-40 minute build (protobuf, abseil, onnx, and ONNX Runtime
itself, all from source). If you only need to run `onnx-finetune` against
artifacts someone else generated, the second (`--enable_training_apis`) build
is sufficient and noticeably smaller in scope than the first.

If `pip install .` on the wheel build fails with a `setuptools`/`distutils`
`install_layout` error, that's an unrelated packaging-step bug in older
`setup.py` against newer `setuptools` -- the compiled Python module already
exists by that point, at `<build_dir>/Release/build/lib/`, and can be used
directly by pointing `PYTHONPATH` there without needing an installed wheel:

```sh
PYTHONPATH=/path/to/onnxruntime/build/Release/build/lib \
  python3 scripts/generate_artifacts.py ...
```

## Building onnx-finetune

```sh
cmake -B build \
  -DORT_SOURCE_DIR=/path/to/onnxruntime \
  -DORT_BUILD_DIR=/path/to/onnxruntime/build/Release
cmake --build build
```

## Usage

```sh
# 1. Generate training artifacts for your model (offline, once).
python3 scripts/generate_artifacts.py model.onnx -o artifacts --loss mse --optimizer adamw

# Freeze everything except a subset (e.g. only fine-tune a head/adapter --
# this is what keeps memory and compute down on larger models):
python3 scripts/generate_artifacts.py model.onnx -o artifacts \
  --freeze-prefix backbone. --loss cross-entropy --optimizer adamw

# 2. Run the actual training loop.
./build/onnx-finetune \
  --artifacts-dir artifacts \
  --train-input train_input.bin --train-target train_target.bin \
  --input-dim 4 --target-dim 1 --num-samples 2048 \
  --batch-size 32 --epochs 20 --lr 0.01 \
  --output-model finetuned.onnx --output-names output
```

`--train-input`/`--train-target` are raw contiguous `float32` binary files
(`num_samples * dim` values each, row-major) -- deliberately the simplest
possible format so the tool has no dataset-library dependency. Convert
whatever you actually have (images, tokenized text, ...) to that layout
before calling this; see `scripts/make_synthetic_data.py` for a minimal
example of writing the format from numpy.

`finetuned.onnx` is a normal inference-only ONNX model afterward -- load it
with any ONNX Runtime build (including a plain `pip install onnxruntime`,
verified in testing), `onnxsim` it, ship it, whatever you'd normally do.

## Try it end-to-end with the included toy example

```sh
python3 scripts/make_toy_model.py -o toy_model.onnx
python3 scripts/make_synthetic_data.py --num-samples 2048
python3 scripts/generate_artifacts.py toy_model.onnx -o artifacts
./build/onnx-finetune \
  --artifacts-dir artifacts \
  --train-input train_input.bin --train-target train_target.bin \
  --input-dim 4 --target-dim 1 --num-samples 2048 \
  --batch-size 32 --epochs 20 --lr 0.01 \
  --output-model finetuned.onnx --output-names output
```

The toy task is learning `y = sum(x)` with a 2-layer MLP; loss should drop
from roughly 5 to under 0.001 over the 20 epochs.

## LoRA (low-rank adaptation)

`--freeze-prefix` above still trains full-size parameter tensors, just fewer
of them. For real LoRA -- freezing every original weight and training only a
pair of small rank-decomposition matrices alongside each targeted layer --
use `scripts/inject_lora.py` to do the graph surgery first, then point
`generate_artifacts.py` at the resulting adapter manifest:

```sh
# 1. Graft a rank-4 adapter onto every eligible MatMul/Gemm weight (or use
#    --target-contains to pick specific layers, e.g. --target-contains q_proj).
python3 scripts/inject_lora.py model.onnx -o model_lora.onnx \
  --rank 4 --params-out lora.json

# 2. Generate training artifacts that train *only* the injected lora_A/lora_B
#    tensors -- everything else (including the original weights) is frozen.
python3 scripts/generate_artifacts.py model_lora.onnx -o artifacts \
  --lora-params-file lora.json --loss cross-entropy --optimizer adamw

# 3. Train as usual.
./build/onnx-finetune --artifacts-dir artifacts ... --output-model finetuned.onnx --output-names output

# 4. Pull just the trained adapter back out (a few KB, not the whole model).
python3 scripts/extract_lora_adapter.py finetuned.onnx --params-file lora.json -o adapter.onnx

# 5. Re-apply that adapter to any fresh copy of the base model, no retraining.
python3 scripts/apply_lora_adapter.py model.onnx --adapter adapter.onnx \
  --params-file lora.json -o finetuned2.onnx
```

`inject_lora.py` adds, per targeted weight `W`, a
`(alpha/rank) * (X @ lora_A @ lora_B)` branch alongside that layer's existing
output (`lora_A` Kaiming-normal, `lora_B` zero, so injecting is a no-op until
trained). Eligible layers are MatMul, Gemm (any `transA`/`transB`), and
1x1/`group=1` Conv (a pointwise conv is just a per-pixel linear layer over
channels) -- see `scripts/lora_surgery.py`'s own docstring for the exact
rule. `adapter.onnx` from step 4 is not a runnable model by itself -- it's a
plain tensor container holding just the `lora_A`/`lora_B` initializers --
meant to be re-merged with `apply_lora_adapter.py`, not loaded directly into
a session. This is a portable, ONNX-native adapter format built on these
scripts; it is not the same binary format as ONNX Runtime GenAI's
`.onnx_adapter` files (those come out of Olive's `convert-adapters`, for
loading multiple adapters into one inference session via
`Ort::LoraAdapter`/`RunOptions`).

### QLoRA (NF4-quantized base + LoRA)

`scripts/prepare_qlora.py` wires the LoRA graph surgery above together with
`onnxsim.nf4.quantize_weight_only_nf4` (bitsandbytes' NF4 4-bit format, see
`onnxsim/nf4.py`) to reproduce Dettmers et al. 2023's actual QLoRA recipe:
freeze a 4-bit-quantized base model and train small full-precision adapters
on top of its on-the-fly dequantized weights, instead of plain LoRA on an
unquantized base. It replaces step 1 above; steps 2-5 are unchanged:

```sh
python3 scripts/prepare_qlora.py model.onnx -o model_qlora.onnx \
  --rank 4 --block-size 64 --params-out lora.json

python3 scripts/generate_artifacts.py model_qlora.onnx -o artifacts \
  --lora-params-file lora.json --loss cross-entropy --optimizer adamw
# ...then train, extract, and re-apply exactly as in the plain-LoRA workflow.
```

This needs the `onnxsim` package itself importable (build/install this
repo's own wheel) for the NF4 quantization step -- unlike every other
onnx-finetune script, which needs only `onnx` + `numpy`. Injection must run
before quantization (quantizing first leaves nothing recognizable for
`inject_lora.py`'s graph surgery to graft onto), and `prepare_qlora.py`
always excludes the newly-injected `lora_A`/`lora_B` weights from
quantization itself -- structurally they're just another small MatMul
against a 2-D initializer, indistinguishable from any other layer's weight
by shape alone, so leaving them eligible would NF4-quantize the adapter too
and defeat the point of keeping it full precision. NF4 quantization only
covers 2-D MatMul/vanilla-Gemm weights (see `onnxsim/nf4.py`), so a 1x1-Conv
LoRA target stays full precision under `prepare_qlora.py` too.

## CLI reference

| flag | required | description |
|---|---|---|
| `--artifacts-dir` | yes | directory containing `checkpoint`, `training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx` |
| `--train-input` / `--train-target` | yes | raw float32 binary files |
| `--input-dim` / `--target-dim` | yes | per-sample element counts |
| `--num-samples` | yes | total samples in the input/target files |
| `--output-model` | yes | where to write the fine-tuned inference `.onnx` |
| `--output-names` | yes | comma-separated graph output name(s) of the *original* model |
| `--batch-size` | no (default 8) | |
| `--epochs` | no (default 10) | |
| `--lr` | no (default 1e-3) | |
| `--save-checkpoint` | no | also save the post-training checkpoint (for resuming training later) |
| `--log-every` | no (default 50) | print loss every N steps; 0 disables |
