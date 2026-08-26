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
| `profile_sample.py` | runs `simplify(path, profile=...)` over named models (via `model_zoo.py`) in isolated subprocesses, for ad hoc investigation of a specific model's profile -- see [Profiling](#profiling). |
| `summarize_profiles.py` | merges per-model `ONNXSIM_PROFILE` traces (written by `worker.py` when profiling is on) into a Markdown summary: which fixed-point span dominates each model, aggregated across the whole sampled set, and (given the regression CSVs too) how much of each model's real wall-clock time the profiler's spans actually cover. See [Profiling](#profiling). |
| `yolov5_regression.py` | standalone check that `onnxsim` can replace the `onnxslim.slim` call in [ultralytics/yolov5](https://github.com/ultralytics/yolov5)'s `export.py`: exports the raw graph, runs both simplifiers, and gates on onnxsim producing a valid graph numerically equivalent to the original. Latest run: [`RESULTS_yolov5.md`](./RESULTS_yolov5.md). |
| `model_zoo.py` | reference a regression model by short name from Python or the CLI, downloading it from the [`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo) org (cached) and returning the path to its main `.onnx`. See [Referencing a model by name](#referencing-a-model-by-name). |

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
  with the largest node-count gaps. Several fusions that onnxslim shipped and
  onnxsim did not — ConvTranspose+BatchNorm and ConvTranspose+Add-bias fusion,
  and no-op `Dropout` elimination in the opset-12+ input form — are now covered
  by onnxsim's optimizer (issue #543) and have regular tests in
  `tests/test_fusion_patterns.py`. A separate onnxsim fix also lets value-baking
  fusions run on IR<4 (e.g. opset-8) exports whose initializers double as graph
  inputs, so plain Conv+BN CNNs like `resnet101-v1-7` now fold. The one fusion
  onnxslim still has that onnxsim lacks is the GELU-subgraph matcher — and
  onnxslim ships it disabled, so both tools currently leave the erf-GELU pattern
  intact (pinned as the remaining `xfail` in `tests/test_fusion_patterns.py`).

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

## Profiling

`run_regression.py --profile-dir DIR` (or the Model Regression workflow's
`profile` `workflow_dispatch` input) captures onnxsim's built-in
`ONNXSIM_PROFILE` trace for every model in that run, one `<model>.json` in
`DIR` per model. It's off by default: a per-model trace adds a background
RSS-sampler thread and a trace write onnxsim otherwise skips, overhead not
worth paying on every scheduled run when nothing's wrong -- turn it on when
investigating a specific slow or regressed run, not routinely.

```bash
# one shard, with a profile trace per model
python scripts/regression/run_regression.py --shard 0 --num-shards 6 \
  --output shard-0.csv --profile-dir profiles

# summarize: which span dominates each model, aggregated across the set, and
# (given the CSV too) how much of each model's real wall-clock time the
# profiler's spans actually cover -- see bench/RESULTS_profiling_survey.md
# for why that gap can be the more important number for a large model.
python scripts/regression/summarize_profiles.py "profiles/*.json" --csv shard-0.csv
```

On the workflow, set `profile: true` on a manual `workflow_dispatch` run;
`regression-profiles-shard-N` / `regression-profiles-slow` artifacts hold the
raw traces (open one in `chrome://tracing` or `ui.perfetto.dev` for the full
flame graph) and a `profile-summary` artifact holds the merged Markdown report.

For an ad hoc look at one model outside the regression set entirely,
`profile_sample.py MODEL_NAME [MODEL_NAME ...]` fetches it via `model_zoo.py`
and prints the same per-span breakdown directly (no CSV, no CI).

### Per-pass profiling (`ONNXSIM_PROFILE_PASS_PHASES`)

`run_regression.py --profile-pass-phases-dir DIR` (or the Model Regression
workflow's `profile_pass_phases` `workflow_dispatch` input) captures onnxsim's
`ONNXSIM_PROFILE_PASS_PHASES` per-optimizer-pass match/modify timing table for
every model in that run, one `<model>.pass_phases.txt` in `DIR` per model --
useful for finding *which single pass* (e.g. `extract_constant_to_initializer`,
see `bench/RESULTS_issue633_followup.md`) dominates a slow or regressed model,
finer-grained than `ONNXSIM_PROFILE`'s per-span trace. It's independent of
`--profile-dir`/`profile` above and much cheaper (two `std::chrono` reads per
node match, no background RSS sampler or trace write), so it's fine to turn on
by itself:

```bash
python scripts/regression/run_regression.py --shard 0 --num-shards 6 \
  --output shard-0.csv --profile-pass-phases-dir pass_phases
```

On the workflow, set `profile_pass_phases: true` on a manual `workflow_dispatch`
run; `regression-pass-phases-shard-N` / `regression-pass-phases-slow`
artifacts hold the raw per-model tables and a `profile-pass-phases` artifact
holds them collected into one Markdown file (no separate summarizer -- the
tables are already onnxsim's own formatted stderr output).

## Referencing a model by name

`model_zoo.py` turns a short model name into a local `.onnx` path, so a script or
test can pull one of the regression models without repeating the
`snapshot_download` boilerplate:

```python
from model_zoo import fetch_model, list_models, resolve

fetch_model("resnet18d_Opset18")   # -> /path/to/…/resnet18d.onnx (cached)
resolve("resnet18d_Opset18")       # -> "onnxmodelzoo/resnet18d_Opset18"
list_models()                      # -> the curated set's full repo ids
```

The name is resolved leniently: a short name is looked up in `models.json`, a
bare name not listed there is assumed to live under `onnxmodelzoo/`, and an
explicit `owner/repo` is used verbatim (so any Hugging Face repo works). The same
is available from the command line:

```bash
python scripts/regression/model_zoo.py list            # curated repo ids
python scripts/regression/model_zoo.py resolve resnet18d_Opset18
python scripts/regression/model_zoo.py fetch resnet18d_Opset18   # prints .onnx path
```

Downloads require `huggingface_hub` (`pip install huggingface_hub`) and land in
its cache, so repeated fetches of the same model are effectively free.

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
