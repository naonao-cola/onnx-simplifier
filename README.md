# ONNX Simplifier

[![PyPI version](https://img.shields.io/pypi/v/onnxsim.svg)](https://pypi.python.org/pypi/onnxsim/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/onnxsim.svg)](https://pypi.python.org/pypi/onnxsim/)
[![PyPI license](https://img.shields.io/pypi/l/onnxsim.svg)](https://pypi.python.org/pypi/onnxsim/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/onnxsim/onnxsim/pulls)
[![Discord](https://img.shields.io/discord/1475920534847099121?logo=discord)](https://discord.gg/W3ht33v4)

_ONNX is great, but sometimes too complicated._

## Background

One day I wanted to export the following simple reshape operation to ONNX:

```python
import torch


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
```

The input shape in this model is static, so what I expected is

![simple_reshape](imgs/simple_reshape.png)

However, I got the following complicated model instead:

![complicated_reshape](imgs/complicated_reshape.png)

## Our solution

ONNX Simplifier is presented to simplify the ONNX model. It infers the whole computation graph
and then replaces the redundant operators with their constant outputs (a.k.a. constant folding).

### Web version

We have published ONNX Simplifier on [GitHub pages](https://onnxsim.github.io/onnxsim/). It works out of the box and **doesn't need any installation**. Note that it runs in the browser locally and your model is completely safe.

### Python version


```
pip3 install -U pip && pip3 install onnxsim
```

Then

```
onnxsim input_onnx_model output_onnx_model
```

For more advanced features, try the following command for help message

```
onnxsim -h
```

## Demonstration

An overall comparison between
[a complicated model](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)
and its simplified version:

![Comparison between old model and new model](imgs/comparison.png)

## In-script workflow

If you would like to embed ONNX simplifier python package in another script, it is just that simple.

```python
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load(filename)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object
```

You can see more details of the API in [onnxsim/onnx_simplifier.py](onnxsim/onnx_simplifier.py)

## Custom operators

Models that contain custom operators, such as TensorRT plugins
(`BatchedNMS_TRT`, `EfficientNMS_TRT`, ...), are supported. onnxsim keeps these
ops unchanged and simplifies the rest of the graph around them. This works
whether the custom op lives in a vendor-specific domain (e.g. `TRT`) or in the
default ONNX domain, so you no longer need to manually move it into a custom
domain to get past validation (issues
[#107](https://github.com/onnxsim/onnxsim/issues/107) and
[#220](https://github.com/onnxsim/onnxsim/issues/220)).

If you describe your custom operator to ONNX with
[`onnx.defs.register_schema`](https://onnx.ai/onnx/api/defs.html), onnxsim
picks that schema up automatically: onnxsim links its own copy of ONNX, so its
operator registry is separate from the `onnx` Python module's, and every
`simplify` call imports the schemas you registered into onnxsim's registry
before validating the model (issue
[#326](https://github.com/onnxsim/onnxsim/issues/326)). You can also trigger the
import explicitly with `onnxsim.import_onnx_schemas()`, or turn the automatic
import off with `onnxsim.simplify(model, import_custom_schemas=False)` (CLI:
`--skip-schema-import`).

```python
import onnx
import onnxsim

# Teach ONNX about your custom operator.
onnx.defs.register_schema(my_op_schema)

# simplify() imports the schema into onnxsim automatically.
model_simp, check_ok = onnxsim.simplify(model)
```

If a registered schema also has a type/shape-inference function (set via
`onnx.defs.OpSchema.set_type_and_shape_inference_function`), onnxsim registers a
trampoline that calls it back through `onnx.shape_inference.infer_node_outputs`
during simplification, so the custom operator's output shapes are inferred too.
Custom operators without an inference function are still imported; shape
inference simply flows past them.

## Changing the opset version

You can upgrade (or downgrade) the model's opset version while simplifying. Pass
`target_opset_version` to `simplify` (CLI: `--target-opset`) and onnxsim converts
the default ONNX domain to that opset — using onnx's own version converter —
before running the simplification, so any redundant nodes the conversion
introduces get cleaned up too.

```python
import onnx
import onnxsim

model = onnx.load(filename)

# Convert the model to opset 18 and simplify it.
model_simp, check = onnxsim.simplify(model, target_opset_version=18)
```

On the command line:

```
onnxsim input_onnx_model output_onnx_model --target-opset 18
```

When `target_opset_version` is left unset (the default), the model's opset
version is preserved.

The conversion runs inside onnxsim's C++ core, so every binding shares it —
the Python package, the C API and its Rust wrapper (`Options::target_opset_version`),
the standalone `onnxsim` binary (`--target-opset`), and the
[web version](https://onnxsim.github.io/onnxsim/) (the "target opset version"
field).

## Custom rewriters

Beyond the built-in optimizer passes, you can plug your own graph rewriting
logic into simplification with the `custom_rewriter` parameter of `simplify()`.
It accepts a callable

```python
Callable[[onnx.ModelProto], Optional[onnx.ModelProto]]
```

that either returns a rewritten model or mutates the model in place and returns
`None`. The callable runs **inside** onnxsim's simplification fixed point,
interleaved with shape inference, the built-in optimizer and constant folding —
so a rewrite can expose new optimization/folding opportunities and vice versa,
and the whole pipeline iterates until it converges. onnxsim itself takes no
dependency on any particular rewriting library; you bring your own.

### Using onnx-rewriter (`onnxscript.rewriter`)

[onnx-rewriter](https://github.com/microsoft/onnxscript) lets you express a
subgraph pattern and its replacement as plain Python and have it matched and
rewritten anywhere in the model. Install it alongside onnxsim:

```
pip3 install onnxscript
```

Then define a rule set and hand it to `simplify` via `custom_rewriter`. This
example fuses `MatMul` + `Add` into a single `Gemm`:

```python
import onnx
import onnxsim
from onnxscript.rewriter import pattern, rewrite

# The subgraph to match: y = MatMul(x, w) + b
def matmul_add_pattern(op, x, w, b):
    return op.Add(op.MatMul(x, w), b)

# What to replace it with: y = Gemm(x, w, b)
def gemm_replacement(op, x, w, b):
    return op.Gemm(x, w, b)

rules = pattern.RewriteRuleSet(
    [pattern.RewriteRule(matmul_add_pattern, gemm_replacement)]
)

model = onnx.load("model.onnx")
model_simp, check = onnxsim.simplify(
    model,
    custom_rewriter=lambda m: rewrite(m, pattern_rewrite_rules=rules),
)
assert check, "Simplified ONNX model could not be validated"
```

Because the rewriter runs every round of the fixed point, the fused `Gemm`
above (and anything it unlocks) is folded and re-optimized together with the
rest of the graph.

A few things to keep in mind:

- **Keep the model schema-valid.** After each rewrite onnxsim validates the
  model, so any op you introduce must be registered at the model's opset (for
  example `Gelu` only exists from opset 20). Custom-domain ops are fine — see
  [Custom operators](#custom-operators) for registering their schemas.
- **Match the opset your rules target.** Convert the model to the opset your
  patterns expect (e.g. with `onnx.version_converter`) before simplifying if
  needed.
- **You are not limited to onnx-rewriter.** Any callable works — a hand-written
  pass over `model.graph`, an [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
  edit, etc. — as long as it takes and returns a `ModelProto`.

## Projects Using ONNX Simplifier

* [MXNet](https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/deploy/export/onnx.html#Simplify-the-exported-ONNX-model)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [ncnn](https://github.com/Tencent/ncnn)
* ...

## Chat

We created a Chinese QQ group for ONNX!

ONNX QQ Group (Chinese): 1021964010, verification code: nndab. Welcome to join!

For English users, I'm active on the [ONNX Slack](https://github.com/onnx/onnx#discuss). You can find and chat with me (daquexian) there.
