# Apple Core ML integration check

Verifies that `onnxsim`'s output still works with **Core ML** — the runtime
behind `coremltools` model deployment on macOS/iOS. The goal is to catch the
failure mode the unit tests and the large-model regression don't: a
simplification that produces a graph Core ML can no longer **compile**, or
that **changes the result** on Apple's stack.

It uses the [`CoreMLExecutionProvider`](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
built into the standard `onnxruntime` PyPI wheel — no extra package, but it
only exists on the **macOS** build (`get_available_providers()` omits it on
Linux/Windows). So the whole check runs on a stock macOS GitHub-hosted
runner with nothing but `pip install onnxruntime`.

## What it checks

For each model the harness runs **original vs. simplified through the same
Core ML backend**, so backend quirks cancel and only an onnxsim-introduced
change can fail the run:

1. `simplify` the model with onnxsim.
2. Compile + run the **original** graph on the Core ML EP.
   If that already fails, the backend just doesn't support the graph →
   reported as `unsupported`, **not** a failure.
3. Compile + run the **simplified** graph on the Core ML EP.
   If the original compiled but the simplified doesn't → `coreml_regression`
   (a failure): simplification broke Core ML compatibility.
4. Compare the two Core ML outputs. Divergence beyond tolerance →
   `coreml_regression`: simplification changed the on-device result.
5. Record the ONNX Runtime CPU-reference diff and the Core ML **coverage**
   (does the whole graph map onto Core ML, or do some nodes fall back to
   ORT's CPU provider) as information.

Partial coverage and `unsupported` are reported, never failed — plenty of
valid graphs are not 100% Core ML-mappable, and that is a backend property,
not an onnxsim bug.

## Files

| file | purpose |
| --- | --- |
| `coreml_backend.py` | wraps the Core ML EP: builds/runs a model on Core ML and on the ORT CPU reference, measures coverage. Degrades gracefully (`COREML_AVAILABLE`) when the EP is absent (non-macOS). |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the small synthetic-graph suite shared with the other EP harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_coreml_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Requires macOS (the platform Core ML itself runs on).

```bash
pip install onnxruntime      # the macOS wheel bundles the Core ML EP
pip install .                # or install an onnxsim wheel

python scripts/apple/run_coreml_compat.py --output coreml-compat.csv
```

The in-tree smoke test `tests/test_coreml_compat.py` reuses this harness and
is skipped automatically when the Core ML EP isn't available (e.g. running on
Linux/Windows).

## Fidelity tiers (what this does and doesn't cover)

This check runs the **real** Core ML compiler and runtime — there is no
emulation step the way QNN's HTP backend needs one, since the check already
runs on real Apple hardware (the macOS CI runner itself). What it leaves
uncovered:

- **Compute unit selection.** `MLComputeUnits=ALL` (the default here) lets
  Core ML place ops on CPU, GPU, or the Neural Engine as it judges best. Set
  `COREML_COMPUTE_UNITS=CPUOnly` (or `CPUAndGPU`, `CPUAndNeuralEngine`) to
  pin a specific target if you need to isolate one.
- **iOS-specific behavior.** This runs the macOS Core ML stack; iOS devices
  share the same compiler but can differ in available ops per OS version.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models can be layered on by passing an on-disk path as
`worker.py`'s second argument, the same way `scripts/qualcomm` and
`scripts/regression` do.

## LLM decode benchmark (`export_llm_to_coreml.py` / `run_llm_decode_benchmark.py`)

A separate pair of tools for a different question than the compatibility
check above: not "does Core ML accept this graph", but "how fast does a
causal LM actually decode through `onnxsim.export_coreml`, on-device" --
the same two axes (decode tok/s, peak memory) as
[DeviceMark](https://devicemark.github.io/)'s on-device LLM leaderboard
methodology.

- `export_llm_to_coreml.py` exports a Hugging Face causal LM (via
  [`optimum-onnx`](https://github.com/huggingface/optimum-onnx)) to an ONNX
  decoder-with-past, runs it through `onnxsim.simplify`, and converts it with
  `onnxsim.export_coreml` using its `dynamic_shapes` argument to keep
  `sequence_length` and `past_sequence_length` genuinely dynamic (bounded by
  `--max-context-length`) instead of baking them to fixed values. The result
  is one Core ML model that supports a real, O(1)-per-token growing KV
  cache: a single forward pass over the whole prompt builds the initial
  cache (prefill), and each new token is generated with a single-token
  forward pass that reuses it, instead of reprocessing the whole context
  every step. This exercises onnxsim's Core ML exporter's dynamic-shape
  support (`onnxsim/coreml_export.py`'s `dynamic_shapes` argument) against
  the largest, most control-flow-heavy transformer graph it's been run on.
- `run_llm_decode_benchmark.py` loads the resulting `.mlpackage`, greedily
  decodes a prompt by prefilling once and then decoding one token at a time
  against the growing cache, and reports prefill latency, decode tok/s
  (decode steps only, matching DeviceMark's methodology), and peak RSS. Like
  the rest of this directory, it only *runs* a model on macOS (that's where
  Core ML's runtime lives); the export step itself needs no Apple hardware.

### Scaling to few-billion-parameter models

DeviceMark's own leaderboard mostly tests models in the 1-4B range, well
past `HuggingFaceTB/SmolLM2-135M-Instruct`'s 135M. The export pipeline above
has also been validated end-to-end against `HuggingFaceTB/SmolLM2-1.7B-Instruct`
(24 layers, ~3.4GB of fp16 weights) -- converting a model at that scale
needs a couple of extra considerations `export_llm_to_coreml.py` handles for
you, and one flag worth knowing about:

- `--dtype fp16` traces and exports the ONNX graph in half precision
  instead of the default float32. Tracing a multi-billion-parameter model in
  float32 can transiently hold more than one full-size copy of its weights
  in memory (PyTorch's own model plus the in-progress ONNX graph); `fp16`
  roughly halves that peak. This is independent of Core ML's own output
  precision, which defaults to float16 regardless of the ONNX input's dtype.
- The batch-size fix-up and `onnxsim.simplify` step both operate on the
  model **by file path**, not as an in-memory `ModelProto` -- passing a
  `ModelProto` to either serializes the whole model to one protobuf message
  first, which protobuf itself caps at 2GiB (comfortably cleared by a
  multi-billion-parameter model's weights). Passing a path instead uses
  onnx's/onnxsim's own file-based C++ entry points, which need only about
  1x the model's size in peak memory rather than 2x+ (see
  `bench/RESULTS_synthetic_decoder_oom.md` in the repo root for the
  investigation that fixed the `onnxsim.simplify` side of this).
- `main_export(..., do_validation=False)` skips `optimum`'s own
  PyTorch-vs-ONNX-Runtime comparison pass, which otherwise keeps a second
  full copy of the model resident purely to check optimum's own export --
  a check this pipeline doesn't rely on (onnxsim's `onnx.checker.check_model`
  and the Core ML conversion actually succeeding are the checks that matter
  here).

```bash
pip install "optimum-onnx" transformers coremltools
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --output smollm2.mlpackage
python run_llm_decode_benchmark.py smollm2.mlpackage \
    --prompt "The capital of France is" --max-new-tokens 20
```
