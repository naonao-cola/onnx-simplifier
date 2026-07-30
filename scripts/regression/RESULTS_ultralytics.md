# ultralytics: onnxsim vs onnxslim (ONNX export regression)

Tests whether **onnxsim** can replace **onnxslim** at the one place ultralytics
uses it — the ONNX export "simplify" step in
[`ultralytics/engine/exporter.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py):

```python
# ultralytics export_onnx(), simplify=True path
import onnxslim
model_onnx = onnxslim.slim(model_onnx)          # incumbent
# candidate:
# import onnxsim
# model_onnx, ok = onnxsim.simplify(model_onnx)
```

`scripts/regression/ultralytics_export_compare.py` exports each YOLO task to a
raw (un-simplified) ONNX graph, then runs **both** simplifiers on that identical
graph and compares node count, ONNX Runtime validity, and numerical parity
(max abs diff of every output vs the raw export on a fixed random input).

## Environment

| package | version |
| --- | --- |
| ultralytics | 8.4.112 |
| onnxsim | 0.7.0 |
| onnxslim | 0.1.94 |
| onnx | 1.22.0 |
| onnxruntime | 1.28.0 |
| torch | 2.13.0+cpu (torchvision 0.28.0+cpu) |
| opset | 20 |

## Results

Models: `yolo11n` across all five tasks, plus a dynamic-axes detect export
(`dynamic=True`) to stress dynamic-shape passes. `raw` = node count of the
un-simplified export; `parity` = max abs output diff vs that raw graph.

| model (task) | raw | onnxsim nodes | onnxsim s | onnxsim parity | onnxslim nodes | onnxslim s | onnxslim parity |
| --- | --: | --: | --: | --: | --: | --: | --: |
| yolo11n (detect) | 355 | **320** | 0.64 | 0.0 | 320 | 1.07 | 0.0 |
| yolo11n-seg (segment) | 393 | **355** | 0.58 | 0.0 | 355 | 0.60 | 0.0 |
| yolo11n-cls (classify) | 153 | **144** | 0.29 | 0.0 | 144 | 0.20 | 0.0 |
| yolo11n-pose (pose) | 405 | **354** | 0.63 | 0.0 | 354 | 0.66 | 0.0 |
| yolo11n-obb (obb) | 385 | **356** | 0.33 | 0.0 | 356 | 0.77 | 0.0 |
| yolo11n (detect, dynamic) | 575 | **419** | 0.70 | 0.0 | 419 | 1.87 | 0.0 |

**Result: onnxsim is a clean drop-in for onnxslim on every tested model.** For
all six configurations the two simplifiers land the **exact same node count**,
both graphs load and run in ONNX Runtime, and both reproduce the raw export's
outputs to the bit (parity `0.0`). onnxsim's own `check_ok` equivalence check
passed in every case. onnxsim was also faster in aggregate (~3.2s vs ~5.2s
total), with the largest gap on the dynamic-shape export (0.70s vs 1.87s) —
historically the case that stressed onnxsim's C++ passes.

## Reproduce

```bash
pip install ultralytics onnxsim onnxslim onnxruntime "onnx>=1.12,<2"
python scripts/regression/ultralytics_export_compare.py --dynamic-detect
```

Weights auto-download from the ultralytics GitHub release assets. Exit code is
non-zero if onnxsim fails to produce a valid, numerically-matching graph for any
model.

## Notes

- One environment snag, unrelated to either simplifier: a CPU-only `torch`
  paired with a non-`+cpu` `torchvision` wheel makes `yolo11n-cls.pt` fail to
  load with `operator torchvision::nms does not exist`. Installing the matching
  `torchvision==<ver>+cpu` build fixes it; classify then exports and passes.
- The dynamic-detect input must use a real spatial size (the harness feeds
  640×640); a degenerate 1×1 map breaks the detector's own concat before any
  simplifier runs.
