# Large-model regression

Runs `onnxsim` over a set of real-world models (image classifiers, detectors,
transformers) pulled from the Hugging Face [`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo)
org, to catch failures that the unit tests don't reach — crashes, C++ aborts
from optimizer passes on dynamic-shape graphs, hangs, and correctness-check
regressions. This complements `tests/`, which uses small synthetic graphs.

Each model is **also run through [`onnxslim`](https://github.com/inisis/OnnxSlim)**
on the same graph, purely for comparison. `onnxsim` is the only tool that gates
the run; the onnxslim numbers just tell us where the two simplifiers diverge on
robustness, optimization strength, and speed / memory (see
[onnxsim vs onnxslim](#onnxsim-vs-onnxslim) below).

Driven by the [`Model Regression`](../../.github/workflows/model-regression.yml)
workflow on a weekly schedule and on demand.

## Files

| file | purpose |
| --- | --- |
| `models.json` | the model set, with per-model baseline timing (for shard balancing) and baseline node counts (for the summary delta). `known_slow: true` marks models that exceeded the standard cap. |
| `worker.py` | downloads one model, then runs **both** onnxsim and onnxslim over it, **each in its own child subprocess** with its own timeout (so a hang/abort in one tool is contained and can't corrupt the other's result), records both outcomes + per-tool peak RSS, and deletes the download. |
| `run_regression.py` | assigns a balanced shard of the model set (or the known-slow set) and runs each model through `worker.py` with a per-tool timeout. Exits non-zero if any model in the shard **crashed, timed out, or failed onnxsim's correctness check** — onnxslim outcomes are recorded but never affect the exit code. |
| `summarize.py` | merges the per-shard CSVs into `regression-report.csv` and a Markdown run summary, including the onnxsim-vs-onnxslim comparison tables. |

## What counts as a failure

A model fails the regression when **onnxsim** **crashes**, **times out**, or when
the simplified graph **does not pass onnxsim's own correctness check**. onnxslim
is comparison-only and **never** fails the run. Node-count drift versus the
baseline is reported but does not fail the run — legitimate optimizer changes
move those numbers.

If a specific optimizer pass raises a C++ assertion, `worker.py` records that
pass, skips it, and retries, so a newly-fragile pass shows up in the summary's
"passes skipped" table instead of silently passing.

## onnxsim vs onnxslim

The summary's comparison section is built from the per-tool CSV columns
(`slim_status`, `slim_simp_nodes`, `slim_seconds`, `slim_valid`,
`slim_peak_rss_mb`, …). It reports three axes, matching the known causes of
divergence:

- **Robustness / failures.** A per-tool status breakdown plus the list of models
  where only one tool produced a usable graph. The two tools have different
  failure modes: onnxsim's optimizer passes are C++ and some still *abort* on
  dynamic-shape graphs (the harness detects the pass, skips it, and retries —
  see the "passes skipped" table); onnxslim's passes are pure Python, so it
  tends to fail differently (or not at all) on the same graphs. A concrete case:
  NVIDIA ModelOpt switched its quantization preprocessing from onnxsim to
  onnxslim in part because onnxsim aborted on ModelOpt's fp8 QDQ output with
  `no supported data type: 17` (onnxsim issue #348; now covered by
  `tests/test_simple.py::test_fp8_qdq_modelopt_integration`).

- **Optimization strength.** Median node reduction for each tool and the models
  with the largest node-count gaps. onnxslim ships pattern fusions onnxsim
  (onnxoptimizer + constant folding) does not — e.g. ConvTranspose+BatchNorm and
  ConvTranspose+Add-bias fusion, and no-op `Dropout(ratio=0)` elimination — so it
  often lands fewer nodes on conv/transformer graphs. Those specific gaps are
  pinned as `xfail` tests in `tests/test_fusion_patterns.py`; they'll XPASS if
  onnxsim ever gains the pass. (onnxslim also has a GELU-subgraph matcher, but it
  ships disabled, so both tools currently leave the erf-GELU pattern intact.)

- **Speed / memory.** Median wall-clock and peak RSS over the models both tools
  completed. onnxsim runs onnxruntime-based constant folding and its `check_n`
  equivalence check, which is where most of its time and memory on large graphs
  goes; onnxslim's optimize step is typically lighter.

The comparison is descriptive, not a target: onnxslim reducing more nodes on a
model is *not* a regression, and is not actionable on its own.

## Running locally

```bash
pip install onnxruntime huggingface_hub onnxslim
pip install .            # or install an onnxsim wheel

# one balanced shard of the blocking set (--timeout is the per-tool cap)
python scripts/regression/run_regression.py --shard 0 --num-shards 6 --output shard-0.csv

# the known-slow models (large / slow to simplify)
python scripts/regression/run_regression.py --slow-only --timeout 2400 --output slow.csv

# combine into a report (writes the onnxsim-vs-onnxslim comparison too)
python scripts/regression/summarize.py "shard-*.csv" slow.csv
```

Set `SLIM_CHECK=0` to skip onnxslim's (optional) equivalence check, which halves
onnxslim's per-model work when you only care about its robustness and node counts.

## Updating the model set

`models.json` is a curated spread across architecture families, not every
export in the org. Edit it by hand for one-off tweaks, or use
`select_models.py` to regenerate/extend it from the Hugging Face
[`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo) org:

```bash
# family coverage: what the org has vs what the set represents (no changes)
python scripts/regression/select_models.py

# bump every entry to the newest Opset export of the same model
python scripts/regression/select_models.py --refresh --write

# add widely-used NLP transformers (one newest-opset representative each)
python scripts/regression/select_models.py \
  --add '^(distilbert|roberta|albert|electra|deberta|bart|xlnet|mobilebert|mpnet|longformer|gpt2|opt)_Opset' --write
```

It preserves existing `baseline_*` and `known_slow` across a regenerate and
gives new entries null baselines. Offline, pass `--ids-file <cached.json>` (a
JSON list of `onnxmodelzoo/<name>`) instead of hitting the API.

Field notes: `baseline_seconds` only affects how models are distributed across
shards; a rough value is fine (new entries default to 0 and land in the
lightest shard until a run fills them in). Set `known_slow: true` for anything
that regularly exceeds ~15 min so it stays out of the blocking shards. After
adding models, run `run_regression.py` once to populate baselines and confirm
they pass before relying on them.
