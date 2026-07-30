# YOLO × onnxsim compatibility harness

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) is the most
widely deployed real-time object-detection family. Its ONNX export path has
long shipped a **simplify** step backed by onnxsim (the `simplify=True`
export flag historically shelled out to `onnxsim.simplify`), so onnxsim is a
first-class part of the YOLO → ONNX deployment pipeline.

`simplify_yolo.py` is a standalone compatibility harness. For each spec it
builds the model from its Ultralytics **YAML config** (random weights, no
checkpoint download), exports it with `YOLO.export(format="onnx")`, and runs
the **current** onnxsim on the result with `check_n=3` — reproducing the
step Ultralytics' own export performs. It reports node counts before/after
and whether onnxsim's numerical equivalence check passed.

Building from the YAML config (rather than a pretrained `.pt`) keeps the
harness fast and fully offline: the weight *values* don't affect the graph
onnxsim simplifies, so the structural coverage is identical to the shipped
models.

## Regression test

`tests/test_yolo.py` is the automated counterpart. It builds four of the
newest specs offline — `yolo26n` (detect), `yolo26n-seg` (segment),
`yolo12n` (detect) and `yolo11n-pose` (pose) — exports each with
Ultralytics, runs it through onnxsim's `check_n=3` verification, and then
loads the simplified graph in onnxruntime to confirm it still runs and keeps
the export's I/O signature. An onnxsim regression on YOLO-style graphs fails
the suite.

`ultralytics` is a large dependency (it pulls torch) and is **not** a normal
test requirement, so the test `importorskip`s it and skips unless it is
already installed. To run it locally::

    pip install ultralytics onnxruntime
    pip install --force-reinstall --no-deps .   # the onnxsim under test
    pytest tests/test_yolo.py -v

## Running

```bash
pip install ultralytics onnxruntime onnxsim
python scripts/yolo/simplify_yolo.py --all      # newest gens × every task head
python scripts/yolo/simplify_yolo.py yolo26n yolo12n-seg   # specific specs
```

`onnxruntime` is **not** optional: onnxsim uses it to numerically compare the
original and simplified graphs (`check_n`). Without it, onnxsim falls back to
onnx's slower Python reference evaluator.

## Notes

- **What onnxsim removes.** YOLO exports carry a lot of foldable anchor-grid
  construction and shape arithmetic in the detection head, plus Conv+BatchNorm
  pairs in the backbone. onnxsim's constant folding + BN-into-Conv fusion
  removes ~22–36% of nodes depending on generation and head.
- **Coverage.** `--all` covers YOLO26 / YOLO12 / YOLO11 × detect, `-seg`,
  `-pose`, `-obb`, `-cls`. `simplify_yolo.py` also accepts the YOLO26-only
  heads `yolo26n-depth`, `yolo26n-sem`, and the higher-resolution `-p2` /
  `-p6` detect variants.
- **Ultralytics now bundles onnxslim**, not onnxsim, for its own
  `simplify=True` path; this harness confirms the exported graphs still
  simplify cleanly and equivalently under onnxsim.

See [`RESULTS.md`](RESULTS.md) for a captured run.
