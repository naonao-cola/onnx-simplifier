# Paddle2ONNX → onnxsim regression

Answers one question: **does onnxsim currently simplify the ONNX graphs that
[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) emits, and does the
result stay numerically equivalent?**

The question is not hypothetical. Paddle2ONNX's `convert.py` has an onnxsim
optimization path, but it is **commented out** upstream — the block guarded by
`optimize_tool == "onnxsim"` at
[`convert.py#L267`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/187729275c36be251a9c676d4ff5550174c10189/paddle2onnx/convert.py#L267):

```python
# if optimize_tool == "onnxsim":
#     ...
#     from onnxsim import simplify
#     onnx_model = onnx.load_model(model_stream)   # produced by paddle2onnx.export
#     simplified_model, check = simplify(onnx_model)
#     ...
```

This harness re-runs exactly that path on a spread of real PaddlePaddle models so
we can track whether onnxsim is in shape to be re-enabled there.

## What it does

For each model, in its own child process (so a C++ abort or hang in one graph
can't take the run down — same isolation as [`../worker.py`](../worker.py)):

1. build a real `paddle.vision.models` network and `paddle.jit.save` it to a
   Paddle inference model (`.json` / `.pdiparams`);
2. convert it with `paddle2onnx.export(..., save_file=None)` — the same call
   `convert.py` makes to get `onnx_model_str`;
3. run `onnxsim.simplify` and keep its correctness flag (`check`);
4. independently re-verify by running the pre- and post-simplify graphs through
   onnxruntime on shared random inputs.

A model **fails** if the paddle2onnx export fails, onnxsim crashes / times out /
returns `check=False`, or the onnxruntime outputs diverge.

## Running

```bash
pip install onnx onnxruntime onnxsim paddlepaddle paddle2onnx
python run_paddle2onnx_regression.py                       # full set
python run_paddle2onnx_regression.py --only resnet18 mobilenet_v2
```

Writes `paddle2onnx-regression.csv` and `paddle2onnx-regression.md`.

## Current support status

**onnxsim v0.7.0 simplifies every Paddle2ONNX-exported model in the set
correctly — 17/17 pass.** Every graph passes both onnxsim's own `check` and the
independent onnxruntime equivalence check, with 66–83 % node reduction. The two
dynamic-batch graphs — the shape regime where onnxsim's C++ passes have
historically aborted — pass as well. There is no onnxsim-side blocker to
re-enabling the commented-out `optimize_tool == "onnxsim"` path in Paddle2ONNX.

Full run captured in [`results-onnxsim-0.7.0.csv`](results-onnxsim-0.7.0.csv)
(onnxsim v0.7.0, paddle2onnx 2.1.0, paddlepaddle 3.3.1, onnx 1.17.0,
onnxruntime 1.28.0, opset 11).

| model | orig→simp nodes | reduction | onnxsim check | ort equiv |
| --- | --- | --- | --- | --- |
| alexnet | 82→21 | 74.4% | ✅ | ✅ |
| vgg16 | 158→38 | 75.9% | ✅ | ✅ |
| resnet18 | 190→49 | 74.2% | ✅ | ✅ |
| resnet50 | 469→122 | 74.0% | ✅ | ✅ |
| resnext50_32x4d | 469→122 | 74.0% | ✅ | ✅ |
| wide_resnet50_2 | 469→122 | 74.0% | ✅ | ✅ |
| mobilenet_v1 | 231→57 | 75.3% | ✅ | ✅ |
| mobilenet_v2 | 576→100 | 82.6% | ✅ | ✅ |
| mobilenet_v3_small | 513→141 | 72.5% | ✅ | ✅ |
| mobilenet_v3_large | 593→161 | 72.8% | ✅ | ✅ |
| squeezenet1_1 | 303→66 | 78.2% | ✅ | ✅ |
| densenet121 | 1395→368 | 73.6% | ✅ | ✅ |
| shufflenet_v2_x1_0 | 735→174 | 76.3% | ✅ | ✅ |
| googlenet | 311→105 | 66.2% | ✅ | ✅ |
| inception_v3 | 936→216 | 76.9% | ✅ | ✅ |
| resnet18_dynbatch | 190→53 | 72.1% | ✅ | ✅ |
| mobilenet_v2_dynbatch | 576→104 | 81.9% | ✅ | ✅ |
