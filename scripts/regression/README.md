# Large-model regression

Runs `onnxsim` over a set of real-world models (image classifiers, detectors,
transformers) pulled from the Hugging Face [`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo)
org, to catch failures that the unit tests don't reach — crashes, C++ aborts
from optimizer passes on dynamic-shape graphs, hangs, and correctness-check
regressions. This complements `tests/`, which uses small synthetic graphs.

Driven by the [`Model Regression`](../../.github/workflows/model-regression.yml)
workflow on a weekly schedule and on demand.

## Files

| file | purpose |
| --- | --- |
| `models.json` | the model set, with per-model baseline timing (for shard balancing) and baseline node counts (for the summary delta). `known_slow: true` marks models that exceeded the standard cap. |
| `worker.py` | simplifies one model in an isolated subprocess; downloads it, runs `simplify`, records the result, deletes the download. |
| `run_regression.py` | assigns a balanced shard of the model set (or the known-slow set) and runs each model through `worker.py` with a per-model timeout. Exits non-zero if any model in the shard crashed, timed out, or failed onnxsim's correctness check. |
| `summarize.py` | merges the per-shard CSVs into `regression-report.csv` and a Markdown run summary. |

## What counts as a failure

A model fails the regression when it **crashes**, **times out**, or when the
simplified graph **does not pass onnxsim's own correctness check**. Node-count
drift versus the baseline is reported but does not fail the run — legitimate
optimizer changes move those numbers.

If a specific optimizer pass raises a C++ assertion, `worker.py` records that
pass, skips it, and retries, so a newly-fragile pass shows up in the summary's
"passes skipped" table instead of silently passing.

## Running locally

```bash
pip install onnxruntime huggingface_hub
pip install .            # or install an onnxsim wheel

# one balanced shard of the blocking set
python scripts/regression/run_regression.py --shard 0 --num-shards 6 --output shard-0.csv

# the known-slow models (large / slow to simplify)
python scripts/regression/run_regression.py --slow-only --timeout 2400 --output slow.csv

# combine into a report
python scripts/regression/summarize.py "shard-*.csv" slow.csv
```

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
