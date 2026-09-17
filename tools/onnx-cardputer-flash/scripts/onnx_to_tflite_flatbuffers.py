"""Convert a fake-quant "sandwich" ONNX model straight to a `.tflite`
FlatBuffers file, using only the `flatbuffers` package -- no TensorFlow,
no onnx2tf, no local venv juggling.

Why this exists / what it actually does
----------------------------------------
`onnx_to_tflite_micro.py` (this directory's other converter) wraps
onnx2tf, which needs a real TensorFlow install to do the ONNX -> TF ->
TFLite conversion. TensorFlow has no WASM/Pyodide build and is a much
bigger dependency than this repo's other conversion tools carry (nncase,
for comparison, is a plain compiler with no ML-framework dependency --
see ../../onnx-k210-flash/web/ncc/README.md), so that path can't run
in-browser.

This script takes a different, narrower path: it does NOT reimplement
ONNX -> TFLite conversion in general. It recognizes one specific, real
pattern -- the "fake-quant sandwich" that TF->ONNX exporters produce when
re-exporting an *already quantized* TFLite graph (DequantizeLinear -> a
float op -> QuantizeLinear, chained through Reshape/Transpose nodes that
exist only because ONNX's Conv is NCHW and MatMul needs exact 2D shapes,
neither of which TFLite's own ops require) -- and repackages the real
quantized weights, biases, and per-tensor scale/zero-point values *already
present in the ONNX file* into native TFLite tensors and ops. It does not
compute new quantization parameters or requantize anything: every number
this emits came from the ONNX graph verbatim. That is what makes this
tractable without TensorFlow -- the hard part (training-time calibration)
already happened upstream, before this script ever sees the model.

Verified against a real Hugging Face model this way (see
tests/test_onnx_to_tflite_flatbuffers.py and this directory's own
README.md's "Pipeline and what's verified" table / "Not done yet /
follow-ups" for the full writeup):
`ketiswp/tensorflow-Micro-Speech-TinyConv-SpeechCommands-uint8-onnx`'s
`model.onnx`, whose real op graph (confirmed by reading its own
`source/model.tflite` with these same vendored schema bindings) is just
DEPTHWISE_CONV_2D -> FULLY_CONNECTED -> SOFTMAX. The emitted `.tflite`
loads in a real TFLite interpreter and matches
`onnx.reference.ReferenceEvaluator`'s output exactly in 199/200 random
trials (the one exception differs by +-1 -- a rounding-domain difference
between ONNX's reference evaluator and TFLite's fixed-point kernels, not
a bug here; both consumed the identical scale/zero-point/weight bytes
from the ONNX file).

Recognized pattern (see convert() below for the exact node walk):
  - A `Conv` node with `group == 1` and a weight whose input-channel dim is
    1 is emitted as DEPTHWISE_CONV_2D (depth_multiplier = output channels)
    -- this is what the reference model above actually uses, confirmed by
    reading it, not assumed; a `Conv` with `group == 1` and input
    channels > 1 is emitted as plain CONV_2D. Both read their
    activation/weight/bias from the nearest upstream DequantizeLinear
    (skipping any Reshape/Transpose in between, which only exist to
    satisfy ONNX's NCHW convention) and their output scale/zero-point from
    the nearest downstream QuantizeLinear (same skip).
  - A `MatMul` node (2 inputs, no bias -- if the model has a real
    fully-connected bias as a third input/Add, that's not handled here;
    see "Not done yet / follow-ups" in ../README.md) is emitted as
    FULLY_CONNECTED. TFLite's FullyConnected flattens a multi-dim input
    itself, so no Reshape needs to be (or is) emitted ahead of it.
  - A `Softmax` node maps directly to SOFTMAX (beta = 1.0).
  - `Reshape`/`Transpose` nodes are never emitted -- they're skipped
    entirely while tracing an op's real input/output, exactly because the
    ops above don't need them.

Not a general ONNX importer. A model using other ops, a Conv with
group > 1 that isn't the "1 input channel" depthwise case, or a
FullyConnected with a real bias will raise NotImplementedError rather
than emit something silently wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
import flatbuffers
from tflite_schema import schema_py_generated as tfl

# TFLite BuiltinOperator codes actually used here (see schema_py_generated's
# own BuiltinOperator class for the full list).
_OP_DEPTHWISE_CONV_2D = 4
_OP_CONV_2D = 3
_OP_FULLY_CONNECTED = 9
_OP_SOFTMAX = 25

_PADDING_SAME = 0
_PADDING_VALID = 1
_ACTIVATION_NONE = 0
_ACTIVATION_RELU = 1


class _Graph:
    """Read-only view over an ONNX GraphProto: initializer values and
    producer/consumer lookups, used to trace through Dequantize/Quantize
    sandwiches and skip Reshape/Transpose nodes."""

    def __init__(self, graph: onnx.GraphProto):
        self.graph = graph
        self.initializers = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        self.producer = {}  # tensor name -> NodeProto that outputs it
        self.consumers = {}  # tensor name -> list[NodeProto] that read it
        for node in graph.node:
            for out in node.output:
                self.producer[out] = node
            for inp in node.input:
                self.consumers.setdefault(inp, []).append(node)

    def const(self, name: str) -> np.ndarray:
        return self.initializers[name]

    def skip_back(self, name: str) -> str:
        """Follow a tensor name backward through any chain of
        Reshape/Transpose nodes to the real tensor feeding them."""
        while True:
            node = self.producer.get(name)
            if node is None or node.op_type not in ("Reshape", "Transpose"):
                return name
            name = node.input[0]

    def dequant_source(self, name: str) -> tuple[str, float, int]:
        """`name` must be the output of a DequantizeLinear node (after
        skipping Reshape/Transpose). Returns (raw quantized tensor name --
        itself skip_back'd, scale, zero_point)."""
        name = self.skip_back(name)
        node = self.producer.get(name)
        if node is None or node.op_type != "DequantizeLinear":
            raise NotImplementedError(
                f"expected {name} to be a DequantizeLinear output"
            )
        raw_name, scale_name, zp_name = node.input
        scale = float(self.const(scale_name))
        zero_point = int(self.const(zp_name))
        return self.skip_back(raw_name), scale, zero_point

    def quant_sink(self, name: str) -> tuple[str, float, int]:
        """`name` is a compute op's output. Follows forward through any
        Reshape/Transpose consumers to the QuantizeLinear that quantizes
        it. Returns (quantized tensor name, scale, zero_point)."""
        while True:
            consumers = self.consumers.get(name, [])
            reshapish = [n for n in consumers if n.op_type in ("Reshape", "Transpose")]
            if len(reshapish) == 1 and len(consumers) == 1:
                name = reshapish[0].output[0]
                continue
            break
        quant_nodes = [
            n for n in self.consumers.get(name, []) if n.op_type == "QuantizeLinear"
        ]
        if len(quant_nodes) != 1:
            raise NotImplementedError(
                f"expected exactly one QuantizeLinear consumer of {name}, found {len(quant_nodes)}"
            )
        node = quant_nodes[0]
        _, scale_name, zp_name = node.input
        return node.output[0], float(self.const(scale_name)), int(self.const(zp_name))


class _Builder:
    """Thin wrapper over flatbuffers.Builder + the vendored schema module
    tracking tensors/buffers/operators as they're added, and resolving
    tensor names to indices for operator inputs/outputs."""

    def __init__(self):
        self.fb = flatbuffers.Builder(1024)
        self.buffers: list[bytes | None] = [
            None
        ]  # buffer 0 is the schema's reserved empty buffer
        self.tensors: list[dict] = []  # {name, shape, dtype, buffer, scale, zero_point}
        self.name_to_index: dict[str, int] = {}
        self.opcodes: list[int] = []  # BuiltinOperator values, in first-use order
        self.operators: list[
            dict
        ] = []  # {opcode_index, inputs, outputs, options_type, options}

    def add_buffer(self, data: bytes | None) -> int:
        self.buffers.append(data)
        return len(self.buffers) - 1

    def add_tensor(
        self,
        name: str,
        shape: list[int],
        dtype: int,
        data: bytes | None,
        scale: float | None = None,
        zero_point: int | None = None,
    ) -> int:
        buf_idx = self.add_buffer(data)
        idx = len(self.tensors)
        self.tensors.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "buffer": buf_idx,
                "scale": scale,
                "zero_point": zero_point,
            }
        )
        self.name_to_index[name] = idx
        return idx

    def opcode_index(self, builtin: int) -> int:
        if builtin not in self.opcodes:
            self.opcodes.append(builtin)
        return self.opcodes.index(builtin)

    def add_operator(
        self,
        builtin: int,
        inputs: list[int],
        outputs: list[int],
        options_type: int,
        options: dict,
    ):
        self.operators.append(
            {
                "opcode_index": self.opcode_index(builtin),
                "inputs": inputs,
                "outputs": outputs,
                "options_type": options_type,
                "options": options,
            }
        )

    # -- FlatBuffers serialization --------------------------------------

    def _vector_i32(self, values: list[int]):
        self.fb.StartVector(4, len(values), 4)
        for v in reversed(values):
            self.fb.PrependInt32(v)
        return self.fb.EndVector()

    def _vector_u8(self, data: bytes):
        return self.fb.CreateByteVector(data)

    def _build_buffer(self, data: bytes | None) -> int:
        if data:
            vec = self._vector_u8(data)
            tfl.BufferStart(self.fb)
            tfl.BufferAddData(self.fb, vec)
            return tfl.BufferEnd(self.fb)
        tfl.BufferStart(self.fb)
        return tfl.BufferEnd(self.fb)

    def _build_quantization(self, scale: float, zero_point: int) -> int:
        self.fb.StartVector(4, 1, 4)
        self.fb.PrependFloat32(scale)
        scale_vec = self.fb.EndVector()
        self.fb.StartVector(8, 1, 8)
        self.fb.PrependInt64(zero_point)
        zp_vec = self.fb.EndVector()
        tfl.QuantizationParametersStart(self.fb)
        tfl.QuantizationParametersAddScale(self.fb, scale_vec)
        tfl.QuantizationParametersAddZeroPoint(self.fb, zp_vec)
        return tfl.QuantizationParametersEnd(self.fb)

    def _build_tensor(self, t: dict) -> int:
        name_off = self.fb.CreateString(t["name"])
        shape_off = self._vector_i32(t["shape"])
        quant_off = None
        if t["scale"] is not None:
            quant_off = self._build_quantization(t["scale"], t["zero_point"])
        tfl.TensorStart(self.fb)
        tfl.TensorAddShape(self.fb, shape_off)
        tfl.TensorAddType(self.fb, t["dtype"])
        tfl.TensorAddBuffer(self.fb, t["buffer"])
        tfl.TensorAddName(self.fb, name_off)
        if quant_off is not None:
            tfl.TensorAddQuantization(self.fb, quant_off)
        return tfl.TensorEnd(self.fb)

    def _build_options(self, options_type: int, options: dict) -> int:
        if options_type == tfl.BuiltinOptions.DepthwiseConv2DOptions:
            tfl.DepthwiseConv2DOptionsStart(self.fb)
            tfl.DepthwiseConv2DOptionsAddPadding(self.fb, options["padding"])
            tfl.DepthwiseConv2DOptionsAddStrideW(self.fb, options["stride_w"])
            tfl.DepthwiseConv2DOptionsAddStrideH(self.fb, options["stride_h"])
            tfl.DepthwiseConv2DOptionsAddDepthMultiplier(
                self.fb, options["depth_multiplier"]
            )
            tfl.DepthwiseConv2DOptionsAddFusedActivationFunction(
                self.fb, options["activation"]
            )
            return tfl.DepthwiseConv2DOptionsEnd(self.fb)
        if options_type == tfl.BuiltinOptions.Conv2DOptions:
            tfl.Conv2DOptionsStart(self.fb)
            tfl.Conv2DOptionsAddPadding(self.fb, options["padding"])
            tfl.Conv2DOptionsAddStrideW(self.fb, options["stride_w"])
            tfl.Conv2DOptionsAddStrideH(self.fb, options["stride_h"])
            tfl.Conv2DOptionsAddFusedActivationFunction(self.fb, options["activation"])
            return tfl.Conv2DOptionsEnd(self.fb)
        if options_type == tfl.BuiltinOptions.FullyConnectedOptions:
            tfl.FullyConnectedOptionsStart(self.fb)
            tfl.FullyConnectedOptionsAddFusedActivationFunction(
                self.fb, options["activation"]
            )
            return tfl.FullyConnectedOptionsEnd(self.fb)
        if options_type == tfl.BuiltinOptions.SoftmaxOptions:
            tfl.SoftmaxOptionsStart(self.fb)
            tfl.SoftmaxOptionsAddBeta(self.fb, options["beta"])
            return tfl.SoftmaxOptionsEnd(self.fb)
        raise NotImplementedError(f"unhandled options_type {options_type}")

    def _build_operator(self, op: dict) -> int:
        options_off = (
            self._build_options(op["options_type"], op["options"])
            if op["options"]
            else None
        )
        inputs_off = self._vector_i32(op["inputs"])
        outputs_off = self._vector_i32(op["outputs"])
        tfl.OperatorStart(self.fb)
        tfl.OperatorAddOpcodeIndex(self.fb, op["opcode_index"])
        tfl.OperatorAddInputs(self.fb, inputs_off)
        tfl.OperatorAddOutputs(self.fb, outputs_off)
        if options_off is not None:
            tfl.OperatorAddBuiltinOptionsType(self.fb, op["options_type"])
            tfl.OperatorAddBuiltinOptions(self.fb, options_off)
        return tfl.OperatorEnd(self.fb)

    def finish(self, input_indices: list[int], output_indices: list[int]) -> bytes:
        buffer_offs = [self._build_buffer(b) for b in self.buffers]
        tfl.ModelStartBuffersVector(self.fb, len(buffer_offs))
        for off in reversed(buffer_offs):
            self.fb.PrependUOffsetTRelative(off)
        buffers_vec = self.fb.EndVector()

        opcode_offs = []
        for builtin in self.opcodes:
            tfl.OperatorCodeStart(self.fb)
            tfl.OperatorCodeAddDeprecatedBuiltinCode(self.fb, min(builtin, 127))
            tfl.OperatorCodeAddBuiltinCode(self.fb, builtin)
            tfl.OperatorCodeAddVersion(self.fb, 1)
            opcode_offs.append(tfl.OperatorCodeEnd(self.fb))
        tfl.ModelStartOperatorCodesVector(self.fb, len(opcode_offs))
        for off in reversed(opcode_offs):
            self.fb.PrependUOffsetTRelative(off)
        opcodes_vec = self.fb.EndVector()

        tensor_offs = [self._build_tensor(t) for t in self.tensors]
        operator_offs = [self._build_operator(op) for op in self.operators]

        tfl.SubGraphStartTensorsVector(self.fb, len(tensor_offs))
        for off in reversed(tensor_offs):
            self.fb.PrependUOffsetTRelative(off)
        tensors_vec = self.fb.EndVector()

        tfl.SubGraphStartOperatorsVector(self.fb, len(operator_offs))
        for off in reversed(operator_offs):
            self.fb.PrependUOffsetTRelative(off)
        operators_vec = self.fb.EndVector()

        inputs_vec = self._vector_i32(input_indices)
        outputs_vec = self._vector_i32(output_indices)
        subgraph_name_off = self.fb.CreateString("main")

        tfl.SubGraphStart(self.fb)
        tfl.SubGraphAddTensors(self.fb, tensors_vec)
        tfl.SubGraphAddInputs(self.fb, inputs_vec)
        tfl.SubGraphAddOutputs(self.fb, outputs_vec)
        tfl.SubGraphAddOperators(self.fb, operators_vec)
        tfl.SubGraphAddName(self.fb, subgraph_name_off)
        subgraph_off = tfl.SubGraphEnd(self.fb)

        tfl.ModelStartSubgraphsVector(self.fb, 1)
        self.fb.PrependUOffsetTRelative(subgraph_off)
        subgraphs_vec = self.fb.EndVector()

        description_off = self.fb.CreateString("onnx_to_tflite_flatbuffers.py")

        tfl.ModelStart(self.fb)
        tfl.ModelAddVersion(self.fb, 3)
        tfl.ModelAddOperatorCodes(self.fb, opcodes_vec)
        tfl.ModelAddSubgraphs(self.fb, subgraphs_vec)
        tfl.ModelAddDescription(self.fb, description_off)
        tfl.ModelAddBuffers(self.fb, buffers_vec)
        model_off = tfl.ModelEnd(self.fb)

        self.fb.Finish(model_off, file_identifier=b"TFL3")
        return bytes(self.fb.Output())


def _emit_conv_like(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    act_in, weight_in, *bias_in = node.input
    act_name, act_scale, act_zp = g.dequant_source(act_in)
    weight_name, weight_scale, weight_zp = g.dequant_source(weight_in)
    weight = g.const(weight_name)  # ONNX OIHW: [out_ch, in_ch, kh, kw]
    out_ch, in_ch, kh, kw = weight.shape

    attrs = {a.name: a for a in node.attribute}
    stride_h, stride_w = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
    group = attrs["group"].i if "group" in attrs else 1
    padding = _PADDING_VALID if all(p == 0 for p in pads) else _PADDING_SAME

    if act_name not in b.name_to_index:
        raise NotImplementedError(
            f"activation tensor {act_name!r} not registered yet -- only a single-conv chain is supported"
        )
    act_idx = b.name_to_index[act_name]
    act_shape = b.tensors[act_idx]["shape"]  # NHWC
    in_channels = act_shape[-1]

    out_name, out_scale, out_zp = g.quant_sink(node.output[0])

    if bias_in:
        bias_name, bias_scale, bias_zp = g.dequant_source(bias_in[0])
        bias = g.const(bias_name).astype(np.int32)
        if bias_zp != 0:
            raise NotImplementedError("TFLite bias tensors must be zero_point == 0")
        bias_idx = b.add_tensor(
            bias_name, [out_ch], tfl.TensorType.INT32, bias.tobytes(), bias_scale, 0
        )
    else:
        bias_idx = -1

    if group == 1 and in_ch == 1 and in_channels == 1:
        # Depthwise: TFLite weight layout [1, kh, kw, out_ch] (depth_multiplier = out_ch).
        tflite_weight = (
            weight.reshape(out_ch, kh, kw).transpose(1, 2, 0).reshape(1, kh, kw, out_ch)
        )
        weight_idx = b.add_tensor(
            weight_name,
            [1, kh, kw, out_ch],
            tfl.TensorType.UINT8,
            tflite_weight.astype(np.uint8).tobytes(),
            weight_scale,
            weight_zp,
        )
        out_h = (
            act_shape[1] // stride_h
            if padding == _PADDING_SAME
            else (act_shape[1] - kh) // stride_h + 1
        )
        out_w = (
            act_shape[2] // stride_w
            if padding == _PADDING_SAME
            else (act_shape[2] - kw) // stride_w + 1
        )
        out_idx = b.add_tensor(
            out_name,
            [act_shape[0], out_h, out_w, out_ch],
            tfl.TensorType.UINT8,
            None,
            out_scale,
            out_zp,
        )
        b.add_operator(
            _OP_DEPTHWISE_CONV_2D,
            [act_idx, weight_idx, bias_idx],
            [out_idx],
            tfl.BuiltinOptions.DepthwiseConv2DOptions,
            {
                "padding": padding,
                "stride_w": stride_w,
                "stride_h": stride_h,
                "depth_multiplier": out_ch,
                "activation": _ACTIVATION_RELU if out_zp == 0 else _ACTIVATION_NONE,
            },
        )
    elif group == 1:
        # Regular conv: TFLite weight layout [out_ch, kh, kw, in_ch].
        tflite_weight = weight.transpose(0, 2, 3, 1)
        weight_idx = b.add_tensor(
            weight_name,
            [out_ch, kh, kw, in_ch],
            tfl.TensorType.UINT8,
            tflite_weight.astype(np.uint8).tobytes(),
            weight_scale,
            weight_zp,
        )
        out_h = (
            act_shape[1] // stride_h
            if padding == _PADDING_SAME
            else (act_shape[1] - kh) // stride_h + 1
        )
        out_w = (
            act_shape[2] // stride_w
            if padding == _PADDING_SAME
            else (act_shape[2] - kw) // stride_w + 1
        )
        out_idx = b.add_tensor(
            out_name,
            [act_shape[0], out_h, out_w, out_ch],
            tfl.TensorType.UINT8,
            None,
            out_scale,
            out_zp,
        )
        b.add_operator(
            _OP_CONV_2D,
            [act_idx, weight_idx, bias_idx],
            [out_idx],
            tfl.BuiltinOptions.Conv2DOptions,
            {
                "padding": padding,
                "stride_w": stride_w,
                "stride_h": stride_h,
                "activation": _ACTIVATION_RELU if out_zp == 0 else _ACTIVATION_NONE,
            },
        )
    else:
        raise NotImplementedError(f"group={group} conv (in_ch={in_ch}) not supported")


def _emit_matmul(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    if len(node.input) != 2:
        raise NotImplementedError(
            "MatMul with a bias input isn't supported yet -- see module docstring"
        )
    act_in, weight_in = node.input
    act_name, act_scale, act_zp = g.dequant_source(act_in)
    weight_name, weight_scale, weight_zp = g.dequant_source(weight_in)
    weight = g.const(weight_name)  # ONNX MatMul convention: [in_features, out_features]
    in_features, out_features = weight.shape
    tflite_weight = weight.transpose(
        1, 0
    )  # TFLite FullyConnected: [out_features, in_features]

    if act_name not in b.name_to_index:
        raise NotImplementedError(f"activation tensor {act_name!r} not registered yet")
    act_idx = b.name_to_index[act_name]

    out_name, out_scale, out_zp = g.quant_sink(node.output[0])

    weight_idx = b.add_tensor(
        weight_name,
        [out_features, in_features],
        tfl.TensorType.UINT8,
        tflite_weight.astype(np.uint8).tobytes(),
        weight_scale,
        weight_zp,
    )
    out_idx = b.add_tensor(
        out_name, [1, out_features], tfl.TensorType.UINT8, None, out_scale, out_zp
    )
    b.add_operator(
        _OP_FULLY_CONNECTED,
        [act_idx, weight_idx, -1],
        [out_idx],
        tfl.BuiltinOptions.FullyConnectedOptions,
        {"activation": _ACTIVATION_NONE},
    )


def _emit_softmax(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    act_name, act_scale, act_zp = g.dequant_source(node.input[0])
    if act_name not in b.name_to_index:
        raise NotImplementedError(f"activation tensor {act_name!r} not registered yet")
    act_idx = b.name_to_index[act_name]
    out_name, out_scale, out_zp = g.quant_sink(node.output[0])
    out_shape = b.tensors[act_idx]["shape"]
    out_idx = b.add_tensor(
        out_name, out_shape, tfl.TensorType.UINT8, None, out_scale, out_zp
    )
    b.add_operator(
        _OP_SOFTMAX,
        [act_idx],
        [out_idx],
        tfl.BuiltinOptions.SoftmaxOptions,
        {"beta": 1.0},
    )


def convert(onnx_path: str | Path) -> bytes:
    model = onnx.load(str(onnx_path))
    g = _Graph(model.graph)
    b = _Builder()

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise NotImplementedError(
            "only single-input/single-output graphs are supported"
        )

    onnx_input = model.graph.input[0]
    input_shape = [d.dim_value for d in onnx_input.type.tensor_type.shape.dim]
    input_name = g.skip_back(onnx_input.name)
    # The graph's real (uint8) input is whatever the first DequantizeLinear
    # sandwich's Conv/MatMul/Softmax ends up tracing back to via
    # dequant_source() -- register it directly here using the ONNX input's
    # own declared shape, since it has no producing node to trace from.
    input_scale = input_zp = None
    for node in model.graph.node:
        if (
            node.op_type == "DequantizeLinear"
            and g.skip_back(node.input[0]) == input_name
        ):
            input_scale = float(g.const(node.input[1]))
            input_zp = int(g.const(node.input[2]))
            break
    if input_scale is None:
        raise NotImplementedError(
            "could not find the input tensor's quantization params"
        )
    input_idx = b.add_tensor(
        input_name, input_shape, tfl.TensorType.UINT8, None, input_scale, input_zp
    )

    for node in model.graph.node:
        if node.op_type == "Conv":
            _emit_conv_like(b, g, node)
        elif node.op_type == "MatMul":
            _emit_matmul(b, g, node)
        elif node.op_type == "Softmax":
            _emit_softmax(b, g, node)
        elif node.op_type in (
            "Reshape",
            "Transpose",
            "QuantizeLinear",
            "DequantizeLinear",
        ):
            pass  # handled by the trace helpers above, not emitted directly
        else:
            raise NotImplementedError(f"unsupported op {node.op_type!r}")

    # The graph's declared output is exactly the tensor name the last
    # QuantizeLinear sandwich produced (quant_sink() returns the
    # QuantizeLinear node's own output name), already registered above.
    output_idx = b.name_to_index[model.graph.output[0].name]
    return b.finish([input_idx], [output_idx])


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <model.onnx> <model.tflite>", file=sys.stderr)
        raise SystemExit(1)
    data = convert(sys.argv[1])
    Path(sys.argv[2]).write_bytes(data)
    print(f"wrote {sys.argv[2]} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
