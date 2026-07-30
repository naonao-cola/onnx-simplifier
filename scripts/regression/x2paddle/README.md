# onnxsim → X2Paddle downstream regression

Checks that `onnxsim` doesn't break [**X2Paddle**](https://github.com/PaddlePaddle/X2Paddle)'s
ONNX → PaddlePaddle conversion.

This is a *downstream-consumer* regression, distinct from the sibling
[onnxmodelzoo sweep](../README.md) (which runs onnxsim standalone and compares
it to onnxslim). Here onnxsim isn't a peer — **it's a dependency inside
X2Paddle**: `x2paddle.convert.onnx2paddle`'s default `enable_optim=True` path
runs `onnxsim.simplify` on the graph before the op-mapper sees it. So an onnxsim
change that rewrites a graph into something X2Paddle's op-mapper can no longer
translate is a real, user-visible breakage of X2Paddle — and this harness is
what catches it.

## How it works

For each model in `models.json` (pulled from the Hugging Face
[`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo) org), `worker.py` runs two
arms and compares them:

| arm | what it does |
| --- | --- |
| **baseline** | the *original* ONNX graph → X2Paddle (decoder → op-mapper → optimizer → `gen_model`), onnxsim **off** |
| **onnxsim** | `onnxsim.simplify` first, then the *simplified* graph through the same X2Paddle stages |

Each step (onnxsim, and each conversion) runs in **its own child subprocess**
with its own timeout, for two reasons:

1. onnxsim's optimizer passes are C++ and some still abort on odd graphs; an
   abort/segfault/hang is contained and reported instead of taking the run down.
2. **Paddle's parameter registry is process-global** — converting both arms in
   one interpreter makes the second collide with the first
   (`parameter name [...] have be been used`). Separate processes are the only
   clean fix.

## Verdicts — what fails the run

| verdict | gates? | meaning |
| --- | --- | --- |
| `pass` | — | onnxsim ok and X2Paddle converted the simplified graph |
| `regression` | **✗ fails** | X2Paddle converted the original but **not** the simplified graph — onnxsim broke a working conversion |
| `onnxsim_fail` | **✗ fails** | onnxsim crashed, timed out, or failed its own correctness check |
| `baseline_unsupported` | — | X2Paddle can't convert the original either (an unsupported op, etc.) — onnxsim is not implicated |
| `improved` | — | the original failed but the simplified graph converted — onnxsim *unblocked* X2Paddle |

Only `regression` and `onnxsim_fail` (and a harness `error`) turn the run red.
`baseline_unsupported` is expected for models that use ops X2Paddle's ONNX
front-end doesn't implement; those are recorded for coverage, never gated.

If an onnxsim optimizer pass raises a C++ assertion, the worker detects the
pass, adds it to `skipped_optimizers`, and retries — so a newly-fragile pass
shows up in the summary's "passes skipped" table (same behaviour as the
onnxmodelzoo sweep).

## onnxslim comparison (non-gating)

Each model is **also** run through [`onnxslim`](https://github.com/inisis/OnnxSlim)
and that slimmed graph is fed to X2Paddle too, in its own isolated arm — purely
for comparison, mirroring the onnxmodelzoo sweep. onnxsim is the only arm that
gates the run; the onnxslim numbers never turn it red. The summary reports two
axes:

- **X2Paddle-convertibility.** Over the models X2Paddle converted from the
  original, how often each simplifier's output still converts, and the models
  where the two diverge. This is the axis that matters for this downstream: a
  simplifier that reduces more nodes but produces a graph X2Paddle can't convert
  is *worse* here, not better.
- **Node reduction.** Median reduction for each tool and the largest node-count
  divergences.

The comparison is descriptive, not a target — onnxslim reducing more nodes on a
model is not a regression and is not actionable on its own.

## Running locally

```bash
# X2Paddle 1.6.0 needs onnx<1.16 (it uses onnx.mapping and mis-detects newer
# onnx as "not installed"), and targets opset <= 15. `six` is an undeclared
# x2paddle runtime dep (x2paddle/core/program.py imports it), so install it
# explicitly. onnxslim is the comparison arm.
pip install "paddlepaddle>=2.5" "x2paddle==1.6.0" "onnx<1.16" \
    onnxruntime huggingface_hub onnxslim six
pip install .            # or install an onnxsim wheel

# one shard
python scripts/regression/x2paddle/run_x2paddle_regression.py \
    --shard 0 --num-shards 2 --timeout 600 --output shard-0.csv

# merge shards into a report + Markdown summary
python scripts/regression/x2paddle/summarize.py "shard-*.csv"
```

### Why we drive X2Paddle's stages directly

`worker.py` calls `ONNXDecoder → ONNXOpMapper → GraphOptimizer → gen_model`
itself rather than `x2paddle.convert.onnx2paddle`. That entry point (a) gates on
`onnx.version.version`, which raises on `onnx>=1.16` and makes it silently
return without converting, and (b) always runs onnxsim internally, which would
prevent us from measuring the original-graph baseline. Driving the stages
directly dodges the version gate and lets us turn onnxsim on/off per arm — the
stages themselves are exactly what `onnx2paddle` runs after its checks.

## Updating the model set

`models.json` is a small curated spread of opset-≤15 vision models X2Paddle's
ONNX front-end supports (the exhaustive onnxmodelzoo list lives in the sibling
harness). `baseline_seconds` only balances shards; `baseline_verdict` records
the last observed outcome so a change in convertibility stands out in review.
Add a model by appending `{"id": "onnxmodelzoo/<name>", "baseline_seconds": 6.0}`
and running once to confirm its verdict. Prefer opset ≤ 15 — X2Paddle 1.6.0
rejects newer opsets.
