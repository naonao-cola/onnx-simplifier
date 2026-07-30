# yolov5 regression — onnxsim as an onnxslim replacement

Regression check for whether **onnxsim** can replace **onnxslim** in
[ultralytics/yolov5](https://github.com/ultralytics/yolov5)'s ONNX export.

## Why

yolov5's `export.py::export_onnx` simplifies the exported graph with onnxslim:

```python
# yolov5/export.py
import onnxslim
LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
model_onnx = onnxslim.slim(model_onnx)
```

This regression exports the raw (un-simplified) yolov5 graph and runs it through
**both** simplifiers, checking that onnxsim is a safe drop-in for that
`onnxslim.slim` call: its own correctness check must pass, the result must load
in onnxruntime, and its outputs must match the **original** graph on random
input (rtol/atol `1e-3`). onnxslim is run on the same graphs as the baseline
yolov5 currently ships, purely for comparison.

Reproduce with
[`yolov5_regression.py`](./yolov5_regression.py).

## Result — PASS

**onnxsim is a clean drop-in for onnxslim on yolov5.** On every variant it
produced a graph that is byte-for-byte numerically identical to the original
(`maxdiff = 0.00e+00`), and it lands the **same node count** as onnxslim.

| model | raw nodes | onnxsim | onnxslim | onnxsim equiv? | onnxsim time | onnxslim time |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| yolov5s (static, 640) | 292 | **236** (−19.2%) | 236 (−19.2%) | ✅ maxdiff 0 | 1.2 s | 0.4 s |
| yolov5n (static, 640) | 292 | **236** (−19.2%) | 236 (−19.2%) | ✅ maxdiff 0 | 0.3 s | 0.3 s |
| yolov5s (dynamic, 640) | 544 | **362** (−33.5%) | 362 (−33.5%) | ✅ maxdiff 0 | 1.3 s | 0.9 s |

No per-op-type divergence between the two simplified graphs on any model. The
dynamic export — where onnxsim's C++ optimizer passes have historically been
more fragile than onnxslim's pure-Python ones — simplified without any pass
being skipped or aborting.

### End-to-end pipeline check

Running yolov5's actual `export.py --weights yolov5s.pt --include onnx
--simplify` with `onnxslim.slim` monkeypatched to route through
`onnxsim.simplify` also succeeds end to end:

```
ONNX: starting export with onnx 1.22.0...
ONNX: slimming with onnxslim 0.1.94...
[shim] onnxsim.simplify -> check=True, 292->236 nodes
ONNX: export success ✅ 2.6s, saved as yolov5s.onnx (28.0 MB)
```

The resulting 236-node model loads in onnxruntime (output `output0`).

## Environment

- onnxsim `v0.7.0` (matches this repo's `VERSION`)
- onnxslim `0.1.94`
- onnx `1.22.0`, onnxruntime `1.28.0`
- torch `2.13.0+cpu`, ultralytics `8.4.112`
- yolov5 `master`, weights `yolov5s.pt` / `yolov5n.pt` from release `v7.0`, opset 12
- CPU-only, run 2026-07-30

## Reproduce

```bash
pip install onnx onnxruntime onnxsim onnxslim
# export raw graphs from a yolov5 checkout (needs torch + yolov5 deps), then compare:
python scripts/regression/yolov5_regression.py \
    --export --yolov5 /path/to/yolov5 --weights yolov5s.pt yolov5n.pt

# or, if you already have raw (un-simplified) yolov5 exports:
python scripts/regression/yolov5_regression.py raw_s.onnx raw_n.onnx
```
