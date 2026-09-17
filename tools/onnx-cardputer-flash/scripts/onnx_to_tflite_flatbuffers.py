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
quantized weights, biases, and scale/zero-point values (per-tensor OR
per-channel, both occur -- see PER-CHANNEL QUANTIZATION below)
*already present in the ONNX file* into native TFLite tensors and ops. It
does not compute new quantization parameters or requantize anything:
every number this emits came from the ONNX graph verbatim. That is what
makes this tractable without TensorFlow -- the hard part (training-time
calibration) already happened upstream, before this script ever sees the
model.

Verified against 5 real Hugging Face models this way (all of this
directory's own README candidate-model table; see
tests/test_onnx_to_tflite_flatbuffers.py and that README's "Pipeline and
what's verified" / "Not done yet / follow-ups" for the full writeup):
TinyConv (uint8), DS-CNN, Streaming DS-CNN, DS-CNN Large, and the Deep
Autoencoder (the latter four all int8, not uint8 -- this script reads the
real dtype off each tensor rather than assuming one). Each emitted
`.tflite` loads in a real TFLite interpreter and matches
`onnx.reference.ReferenceEvaluator`'s output on real random inputs (see
that test file for the per-model match rates and the known +-1
rounding-domain caveat, same as before).

Recognized pattern (see convert() below for the exact node walk):
  - `Conv`: `group == 1` is CONV_2D; `group == in_channels` (with
    weight's per-group input-channel dim == 1, i.e. a "real" ONNX
    depthwise conv, not just an unusual `group == 1` case with 1 input
    channel like the uint8 TinyConv model) is DEPTHWISE_CONV_2D
    (depth_multiplier = out_channels / group). Any other group value
    raises NotImplementedError. Bias, when present, is Conv's 3rd input
    (matches every model checked so far) -- see trace_forward() below for
    the case where a compute op's bias instead arrives via a separate
    `Add` node (this repo's Dense/MatMul layers).
  - `MatMul` (bias-less, 2 inputs) and `Gemm` (`transA=0, transB=0`, an
    optional 3rd bias input) both map to FULLY_CONNECTED -- TFLite's
    FullyConnected flattens a multi-dim input itself, so no Reshape needs
    to be (or is) emitted ahead of it. A bias-less MatMul may still pick
    up a bias via the `Add`-node case in trace_forward() (this repo's
    plain Dense layers export this way, unlike Gemm's 3-input form).
  - `AveragePool` maps to AVERAGE_POOL_2D. TFLite's quantized average
    pool operates directly on the quantized values with the *same*
    scale/zero-point in and out (confirmed directly: every real model
    checked reuses the identical QuantizeLinear scale/zero-point right
    before and after the pool's float Dequantize/Quantize pair) -- this
    script relies on that rather than assuming it, since trace_forward()
    finds the real output QuantizeLinear either way.
  - `Softmax` maps directly to SOFTMAX (beta = 1.0).
  - A `Relu` node directly between a compute op's float output and its
    QuantizeLinear becomes that op's `fused_activation_function = RELU`
    (TFLite's real convention -- there is no separate RELU op emitted).
    An `Add` node in the same position, whose other input traces back to
    a DequantizeLinear'd constant, is treated as that op's bias (used by
    this repo's Dense/MatMul layers, which don't take bias as a 3rd
    input the way Conv/Gemm do). Both are recognized by trace_forward().
  - `Reshape`/`Transpose` nodes are never emitted -- they're skipped
    entirely while tracing an op's real input/output, exactly because the
    ops above don't need them.

PER-CHANNEL QUANTIZATION: a DequantizeLinear/QuantizeLinear pair with an
`axis` attribute quantizes per output-channel (a real, common technique
for conv/FC weights -- confirmed directly: every non-TinyConv model
checked uses it for every weight/bias, while every activation stays
per-tensor). This script carries per-channel scale/zero-point arrays
through unchanged and sets TFLite's `quantized_dimension` to match where
the output-channel axis actually ends up after this script's own weight
layout transpose -- 0 for CONV_2D/FULLY_CONNECTED (unchanged from ONNX's
own OIHW axis-0 convention), 3 for DEPTHWISE_CONV_2D (TFLite's `[1, H, W,
C*M]` layout puts the channel axis last).

Not a general ONNX importer. A model using other ops, a Conv `group`
that's neither 1 nor a real per-channel depthwise, a Gemm with
transA/transB set, or a pattern trace_forward() doesn't recognize will
raise NotImplementedError rather than emit something silently wrong.
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
_OP_AVERAGE_POOL_2D = 1
_OP_CONV_2D = 3
_OP_DEPTHWISE_CONV_2D = 4
_OP_FULLY_CONNECTED = 9
_OP_SOFTMAX = 25

_PADDING_SAME = 0
_PADDING_VALID = 1
_ACTIVATION_NONE = 0
_ACTIVATION_RELU = 1

_ONNX_TO_TFLITE_DTYPE = {
    np.dtype("uint8"): tfl.TensorType.UINT8,
    np.dtype("int8"): tfl.TensorType.INT8,
}


def _tflite_dtype(arr: np.ndarray) -> int:
    try:
        return _ONNX_TO_TFLITE_DTYPE[arr.dtype]
    except KeyError:
        raise NotImplementedError(f"unsupported quantized dtype {arr.dtype}") from None


class _Graph:
    """Read-only view over an ONNX GraphProto: initializer values and
    producer/consumer lookups, used to trace through Dequantize/Quantize
    sandwiches and skip Reshape/Transpose/Relu/Add(bias) nodes."""

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

    def dequant_source(self, name: str) -> tuple[str, np.ndarray, np.ndarray]:
        """`name` must be the output of a DequantizeLinear node (after
        skipping Reshape/Transpose). Returns (raw quantized tensor name --
        itself skip_back'd, scale array, zero_point array) -- the arrays
        are shape () for a per-tensor sandwich, shape (C,) for per-channel
        (see module docstring)."""
        name = self.skip_back(name)
        node = self.producer.get(name)
        if node is None or node.op_type != "DequantizeLinear":
            raise NotImplementedError(
                f"expected {name} to be a DequantizeLinear output"
            )
        raw_name, scale_name, zp_name = node.input
        return self.skip_back(raw_name), self.const(scale_name), self.const(zp_name)

    def trace_forward(
        self, name: str
    ) -> tuple[str, np.ndarray, np.ndarray, bool, tuple | None]:
        """`name` is a compute op's raw float output. Follows forward
        through, in any combination: an `Add` (bias -- the other operand
        must trace back to a DequantizeLinear'd constant), a `Relu`, and
        any Reshape/Transpose, to the QuantizeLinear that quantizes the
        final result. Returns (quantized tensor name, scale array,
        zero_point array, has_relu, bias) where bias is None or
        (raw_name, scale array, zero_point array)."""
        bias = None
        has_relu = False
        while True:
            consumers = self.consumers.get(name, [])
            if len(consumers) != 1:
                break
            node = consumers[0]
            if node.op_type in ("Reshape", "Transpose"):
                name = node.output[0]
                continue
            if node.op_type == "Relu" and not has_relu:
                has_relu = True
                name = node.output[0]
                continue
            if node.op_type == "Add" and bias is None:
                other = node.input[1] if node.input[0] == name else node.input[0]
                bias = self.dequant_source(other)
                name = node.output[0]
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
        return (
            node.output[0],
            self.const(scale_name),
            self.const(zp_name),
            has_relu,
            bias,
        )


class _Builder:
    """Thin wrapper over flatbuffers.Builder + the vendored schema module
    tracking tensors/buffers/operators as they're added, and resolving
    tensor names to indices for operator inputs/outputs."""

    def __init__(self):
        self.fb = flatbuffers.Builder(1024)
        self.buffers: list[bytes | None] = [
            None
        ]  # buffer 0 is the schema's reserved empty buffer
        self.tensors: list[
            dict
        ] = []  # {name, shape, dtype, buffer, scale, zero_point, quantized_dimension}
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
        scale: np.ndarray | None = None,
        zero_point: np.ndarray | None = None,
        quantized_dimension: int = 0,
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
                "quantized_dimension": quantized_dimension,
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

    def _build_quantization(
        self, scale: np.ndarray, zero_point: np.ndarray, quantized_dimension: int
    ) -> int:
        scale = np.atleast_1d(scale).astype(np.float32)
        zero_point = np.atleast_1d(zero_point).astype(np.int64)
        self.fb.StartVector(4, len(scale), 4)
        for v in reversed(scale.tolist()):
            self.fb.PrependFloat32(v)
        scale_vec = self.fb.EndVector()
        self.fb.StartVector(8, len(zero_point), 8)
        for v in reversed(zero_point.tolist()):
            self.fb.PrependInt64(v)
        zp_vec = self.fb.EndVector()
        tfl.QuantizationParametersStart(self.fb)
        tfl.QuantizationParametersAddScale(self.fb, scale_vec)
        tfl.QuantizationParametersAddZeroPoint(self.fb, zp_vec)
        if len(scale) > 1:
            tfl.QuantizationParametersAddQuantizedDimension(
                self.fb, quantized_dimension
            )
        return tfl.QuantizationParametersEnd(self.fb)

    def _build_tensor(self, t: dict) -> int:
        name_off = self.fb.CreateString(t["name"])
        shape_off = self._vector_i32(t["shape"])
        quant_off = None
        if t["scale"] is not None:
            quant_off = self._build_quantization(
                t["scale"], t["zero_point"], t["quantized_dimension"]
            )
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
        if options_type == tfl.BuiltinOptions.Pool2DOptions:
            tfl.Pool2DOptionsStart(self.fb)
            tfl.Pool2DOptionsAddPadding(self.fb, options["padding"])
            tfl.Pool2DOptionsAddStrideW(self.fb, options["stride_w"])
            tfl.Pool2DOptionsAddStrideH(self.fb, options["stride_h"])
            tfl.Pool2DOptionsAddFilterWidth(self.fb, options["filter_w"])
            tfl.Pool2DOptionsAddFilterHeight(self.fb, options["filter_h"])
            tfl.Pool2DOptionsAddFusedActivationFunction(self.fb, options["activation"])
            return tfl.Pool2DOptionsEnd(self.fb)
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


def _same_pad_out(in_size: int, stride: int) -> int:
    return -(
        -in_size // stride
    )  # ceiling division -- TFLite's real SAME-padding output size


def _valid_pad_out(in_size: int, k: int, stride: int) -> int:
    return (in_size - k) // stride + 1


def _add_bias_tensor(
    b: _Builder,
    g: _Graph,
    bias_name: str,
    bias_scale: np.ndarray,
    bias_zp: np.ndarray,
    out_ch: int,
) -> int:
    if np.any(np.atleast_1d(bias_zp) != 0):
        raise NotImplementedError("TFLite bias tensors must be zero_point == 0")
    bias = g.const(bias_name).astype(np.int32)
    return b.add_tensor(
        bias_name,
        [out_ch],
        tfl.TensorType.INT32,
        bias.tobytes(),
        bias_scale,
        bias_zp,
        quantized_dimension=0,
    )


def _emit_conv_like(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    act_in, weight_in, *bias_in = node.input
    act_name, _, _ = g.dequant_source(act_in)
    weight_name, weight_scale, weight_zp = g.dequant_source(weight_in)
    weight = g.const(weight_name)  # ONNX OIHW: [out_ch, in_ch/group, kh, kw]
    out_ch, in_ch_per_group, kh, kw = weight.shape

    attrs = {a.name: a for a in node.attribute}
    stride_h, stride_w = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
    group = attrs["group"].i if "group" in attrs else 1
    padding = _PADDING_VALID if all(p == 0 for p in pads) else _PADDING_SAME

    if act_name not in b.name_to_index:
        raise NotImplementedError(
            f"activation tensor {act_name!r} not registered yet -- only a single-chain graph is supported"
        )
    act_idx = b.name_to_index[act_name]
    act_shape = b.tensors[act_idx]["shape"]  # NHWC
    in_channels = act_shape[-1]

    out_name, out_scale, out_zp, has_relu, extra_bias = g.trace_forward(node.output[0])
    activation = _ACTIVATION_RELU if has_relu else _ACTIVATION_NONE

    bias_source = g.dequant_source(bias_in[0]) if bias_in else extra_bias
    bias_idx = _add_bias_tensor(b, g, *bias_source, out_ch) if bias_source else -1

    out_dtype = _tflite_dtype(weight)

    if group == in_channels and in_ch_per_group == 1:
        depth_multiplier = out_ch // group
        # TFLite depthwise weight layout: [1, kh, kw, out_ch] (out_ch axis last).
        tflite_weight = (
            weight.reshape(out_ch, kh, kw).transpose(1, 2, 0).reshape(1, kh, kw, out_ch)
        )
        weight_idx = b.add_tensor(
            weight_name,
            [1, kh, kw, out_ch],
            out_dtype,
            tflite_weight.tobytes(),
            weight_scale,
            weight_zp,
            quantized_dimension=3,
        )
        out_h = (
            _same_pad_out(act_shape[1], stride_h)
            if padding == _PADDING_SAME
            else _valid_pad_out(act_shape[1], kh, stride_h)
        )
        out_w = (
            _same_pad_out(act_shape[2], stride_w)
            if padding == _PADDING_SAME
            else _valid_pad_out(act_shape[2], kw, stride_w)
        )
        out_idx = b.add_tensor(
            out_name,
            [act_shape[0], out_h, out_w, out_ch],
            out_dtype,
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
                "depth_multiplier": depth_multiplier,
                "activation": activation,
            },
        )
    elif group == 1:
        # TFLite regular conv weight layout: [out_ch, kh, kw, in_ch] (out_ch axis first, unchanged).
        tflite_weight = weight.transpose(0, 2, 3, 1)
        weight_idx = b.add_tensor(
            weight_name,
            [out_ch, kh, kw, in_ch_per_group],
            out_dtype,
            tflite_weight.tobytes(),
            weight_scale,
            weight_zp,
            quantized_dimension=0,
        )
        out_h = (
            _same_pad_out(act_shape[1], stride_h)
            if padding == _PADDING_SAME
            else _valid_pad_out(act_shape[1], kh, stride_h)
        )
        out_w = (
            _same_pad_out(act_shape[2], stride_w)
            if padding == _PADDING_SAME
            else _valid_pad_out(act_shape[2], kw, stride_w)
        )
        out_idx = b.add_tensor(
            out_name,
            [act_shape[0], out_h, out_w, out_ch],
            out_dtype,
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
                "activation": activation,
            },
        )
    else:
        raise NotImplementedError(
            f"group={group} conv (in_channels={in_channels}, weight in_ch/group={in_ch_per_group}) not supported"
        )


def _emit_avgpool(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    act_name, _, _ = g.dequant_source(node.input[0])
    if act_name not in b.name_to_index:
        raise NotImplementedError(f"activation tensor {act_name!r} not registered yet")
    act_idx = b.name_to_index[act_name]
    act_shape = b.tensors[act_idx]["shape"]

    attrs = {a.name: a for a in node.attribute}
    kh, kw = list(attrs["kernel_shape"].ints)
    stride_h, stride_w = list(attrs["strides"].ints) if "strides" in attrs else [kh, kw]
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
    if any(p != 0 for p in pads):
        raise NotImplementedError("padded AveragePool not supported")

    out_name, out_scale, out_zp, has_relu, _ = g.trace_forward(node.output[0])
    out_h = _valid_pad_out(act_shape[1], kh, stride_h)
    out_w = _valid_pad_out(act_shape[2], kw, stride_w)
    out_dtype = b.tensors[act_idx]["dtype"]
    out_idx = b.add_tensor(
        out_name,
        [act_shape[0], out_h, out_w, act_shape[3]],
        out_dtype,
        None,
        out_scale,
        out_zp,
    )
    b.add_operator(
        _OP_AVERAGE_POOL_2D,
        [act_idx],
        [out_idx],
        tfl.BuiltinOptions.Pool2DOptions,
        {
            "padding": _PADDING_VALID,
            "stride_w": stride_w,
            "stride_h": stride_h,
            "filter_w": kw,
            "filter_h": kh,
            "activation": _ACTIVATION_RELU if has_relu else _ACTIVATION_NONE,
        },
    )


def _emit_fully_connected(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    is_gemm = node.op_type == "Gemm"
    if is_gemm:
        attrs = {a.name: a for a in node.attribute}
        trans_a = attrs["transA"].i if "transA" in attrs else 0
        trans_b = attrs["transB"].i if "transB" in attrs else 0
        if trans_a or trans_b:
            raise NotImplementedError("Gemm with transA/transB set is not supported")

    act_in, weight_in = node.input[0], node.input[1]
    act_name, _, _ = g.dequant_source(act_in)
    weight_name, weight_scale, weight_zp = g.dequant_source(weight_in)
    weight = g.const(
        weight_name
    )  # ONNX MatMul/Gemm(transB=0) convention: [in_features, out_features]
    in_features, out_features = weight.shape
    tflite_weight = weight.transpose(
        1, 0
    )  # TFLite FullyConnected: [out_features, in_features]

    if act_name not in b.name_to_index:
        raise NotImplementedError(f"activation tensor {act_name!r} not registered yet")
    act_idx = b.name_to_index[act_name]

    bias_source = (
        g.dequant_source(node.input[2]) if is_gemm and len(node.input) == 3 else None
    )
    out_name, out_scale, out_zp, has_relu, extra_bias = g.trace_forward(node.output[0])
    bias_source = bias_source or extra_bias
    bias_idx = _add_bias_tensor(b, g, *bias_source, out_features) if bias_source else -1

    out_dtype = _tflite_dtype(weight)
    weight_idx = b.add_tensor(
        weight_name,
        [out_features, in_features],
        out_dtype,
        tflite_weight.tobytes(),
        weight_scale,
        weight_zp,
        quantized_dimension=0,
    )
    out_idx = b.add_tensor(
        out_name, [1, out_features], out_dtype, None, out_scale, out_zp
    )
    b.add_operator(
        _OP_FULLY_CONNECTED,
        [act_idx, weight_idx, bias_idx],
        [out_idx],
        tfl.BuiltinOptions.FullyConnectedOptions,
        {"activation": _ACTIVATION_RELU if has_relu else _ACTIVATION_NONE},
    )


def _emit_softmax(b: _Builder, g: _Graph, node: onnx.NodeProto) -> None:
    act_name, _, _ = g.dequant_source(node.input[0])
    if act_name not in b.name_to_index:
        raise NotImplementedError(f"activation tensor {act_name!r} not registered yet")
    act_idx = b.name_to_index[act_name]
    out_name, out_scale, out_zp, _, _ = g.trace_forward(node.output[0])
    out_shape = b.tensors[act_idx]["shape"]
    out_dtype = b.tensors[act_idx]["dtype"]
    out_idx = b.add_tensor(out_name, out_shape, out_dtype, None, out_scale, out_zp)
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
    # A dynamic/symbolic dim (dim_param set, no dim_value) reads as 0 via
    # dim_value -- pin it to 1, matching onnx-k210-flash/scripts/
    # onnx_to_kmodel.py's own _pin_batch_dim for the same real reason
    # (every model checked here only has this on the batch dim; on-device
    # inference is batch-1 anyway).
    input_shape = [d.dim_value or 1 for d in onnx_input.type.tensor_type.shape.dim]
    # Every op this script emits reads NHWC (see module docstring); some
    # exporters declare the graph's raw input shape as NCHW-with-C==1
    # instead (confirmed directly: one of the 5 models checked declares
    # `[1, 1, 49, 10]`, not `[1, 49, 10, 1]` like the rest, even though its
    # first real Conv needs the same 1-input-channel NHWC shape the others
    # use directly) -- moving a size-1 non-last dim to the end is a
    # byte-identical reshape (nothing to permute when that axis has one
    # element), so it's safe to canonicalize here rather than getting it
    # wrong deeper in the graph. Only applies when dim 1 doesn't already
    # look like NHWC's channel dim.
    if len(input_shape) == 4 and input_shape[1] == 1 and input_shape[-1] != 1:
        input_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
    input_name = g.skip_back(onnx_input.name)
    # The graph's real quantized input is whatever the first DequantizeLinear
    # sandwich ends up tracing back to via dequant_source() -- register it
    # directly here using the ONNX input's own declared shape/dtype, since
    # it has no producing node to trace from.
    input_scale = input_zp = None
    for node in model.graph.node:
        if (
            node.op_type == "DequantizeLinear"
            and g.skip_back(node.input[0]) == input_name
        ):
            input_scale = g.const(node.input[1])
            input_zp = g.const(node.input[2])
            break
    if input_scale is None:
        raise NotImplementedError(
            "could not find the input tensor's quantization params"
        )
    input_dtype = (
        tfl.TensorType.INT8
        if onnx_input.type.tensor_type.elem_type == onnx.TensorProto.INT8
        else tfl.TensorType.UINT8
    )
    input_idx = b.add_tensor(
        input_name, input_shape, input_dtype, None, input_scale, input_zp
    )

    for node in model.graph.node:
        if node.op_type == "Conv":
            _emit_conv_like(b, g, node)
        elif node.op_type in ("MatMul", "Gemm"):
            _emit_fully_connected(b, g, node)
        elif node.op_type == "AveragePool":
            _emit_avgpool(b, g, node)
        elif node.op_type == "Softmax":
            _emit_softmax(b, g, node)
        elif node.op_type in (
            "Reshape",
            "Transpose",
            "Relu",
            "Add",
            "QuantizeLinear",
            "DequantizeLinear",
        ):
            pass  # handled by the trace helpers above, not emitted directly
        else:
            raise NotImplementedError(f"unsupported op {node.op_type!r}")

    # The graph's declared output is exactly the tensor name the last
    # QuantizeLinear sandwich produced (trace_forward() returns the
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
