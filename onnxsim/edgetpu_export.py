"""Google Coral Edge TPU support for onnxsim's TensorFlow Lite export.

The Edge TPU only runs TensorFlow Lite models that are fully 8-bit quantized
and then compiled specifically for it with the Edge TPU compiler
(``edgetpu_compiler``). This module covers that whole tail of the pipeline,
starting from the (simplified) ONNX model onnxsim produces:

1.  **Compatibility check** (:func:`check_onnx_for_edgetpu`) -- statically
    inspect the ONNX graph for the model requirements the Edge TPU imposes
    (static shapes, rank/size limits, supported ops) *before* paying for a
    conversion. Needs nothing but ``onnx`` itself.
2.  **Full-integer quantization** (:func:`quantize_for_edgetpu`) -- convert to
    ``.tflite`` with integer-only quantization and quantized I/O, via
    :func:`onnxsim.tflite_export.convert_to_tflite` (TensorFlow is required).
3.  **Compilation** (:func:`compile_for_edgetpu`) -- run ``edgetpu_compiler``
    on the quantized model and parse its operator log into which ops landed on
    the TPU and which fell back to the CPU. Only needs the compiler binary.
4.  **Inference** (:func:`make_litert_interpreter`, :func:`run_litert`) -- run
    a model through `LiteRT <https://ai.google.dev/edge/litert>`_
    (``ai_edge_litert``, the successor to ``tflite_runtime``), optionally with
    the Edge TPU delegate (``libedgetpu``) for on-device execution.

:func:`export_edgetpu` chains steps 2-3 into one call.

Like onnxruntime for constant folding, every dependency here is optional and
imported lazily: ``onnx``-only checking always works, while quantization needs
``tensorflow``, model inspection needs ``ai_edge_litert``, compilation needs
the ``edgetpu_compiler`` binary, and on-device inference needs ``libedgetpu``
plus a plugged-in Coral USB Accelerator / PCIe device with readable USB nodes
(see :func:`edgetpu_setup_hint` when the delegate fails to load).
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnx

_LITERT_INSTALL_HINT = (
    "LiteRT is required here but is not installed. "
    "Install it with `pip install ai-edge-litert` "
    "(Google's successor to the deprecated `tflite-runtime` package; "
    "it supports current Python versions, unlike `pycoral`)."
)
_EDGETPU_COMPILER_HINT = (
    "The Edge TPU compiler is required here but was not found. Install it "
    "with `sudo apt-get install edgetpu-compiler` (see "
    "https://coral.ai/docs/edgetpu/compiler/), or pass compiler= with an "
    "explicit path to the `edgetpu_compiler` binary."
)


def has_litert() -> bool:
    """Whether LiteRT (``ai_edge_litert``) is importable in this environment."""
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        return False
    return True


def _import_litert():
    try:
        from ai_edge_litert import interpreter as litert_interpreter
    except ImportError as exc:
        raise RuntimeError(_LITERT_INSTALL_HINT) from exc
    return litert_interpreter


def find_edgetpu_compiler(compiler: Optional[str] = None) -> Optional[str]:
    """Locate the ``edgetpu_compiler`` binary: ``compiler`` if given, else the
    ``EDGETPU_COMPILER`` environment variable, else ``PATH``. Returns ``None``
    when it cannot be found."""
    if compiler:
        return compiler
    env = os.environ.get("EDGETPU_COMPILER")
    if env:
        return env
    return shutil.which("edgetpu_compiler")


def edgetpu_compiler_version(compiler: Optional[str] = None) -> str:
    """Run ``edgetpu_compiler --version`` and return its output line."""
    path = find_edgetpu_compiler(compiler)
    if path is None:
        raise RuntimeError(_EDGETPU_COMPILER_HINT)
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Failed to run the Edge TPU compiler at {path!r}: {exc}"
        ) from exc
    text = (out.stdout or "") + (out.stderr or "")
    for line in text.splitlines():
        if "Edge TPU Compiler version" in line:
            return line.strip()
    raise RuntimeError(f"Unexpected output from `{path} --version`: {text.strip()!r}")


def find_edgetpu_library(library: Optional[str] = None) -> Optional[str]:
    """Locate ``libedgetpu``: ``library`` if given, else the ``EDGETPU_LIBRARY``
    environment variable, else the usual system library locations. Returns
    ``None`` when it cannot be found (it ships with ``libedgetpu1-std`` /
    ``libedgetpu1-max`` -- see :func:`edgetpu_setup_hint`)."""
    if library:
        return library
    env = os.environ.get("EDGETPU_LIBRARY")
    if env:
        return env
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libedgetpu.so.1",
        "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1",
        "/usr/lib/arm-linux-gnueabihf/libedgetpu.so.1",
        "/usr/lib/libedgetpu.so.1",
        "/usr/local/lib/libedgetpu.so.1",
    ]
    for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if path:
            candidates.append(os.path.join(path, "libedgetpu.so.1"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def edgetpu_setup_hint() -> str:
    """Human-readable setup checklist for on-device Edge TPU inference."""
    return (
        "To run a model on a Coral Edge TPU device:\n"
        "  1. Install the runtime: `sudo apt-get install libedgetpu1-std`\n"
        "     (or `libedgetpu1-max` for maximum clock frequency) from\n"
        "     https://coral.ai/docs/accelerator/get-started/ .\n"
        "  2. Make the USB device accessible: the udev rule installed with the\n"
        "     runtime grants access; without it, add\n"
        "     /etc/udev/rules.d/99-edgetpu.rules containing e.g.\n"
        '       SUBSYSTEM=="usb", ATTR{idVendor}=="1a6e", MODE="0666", GROUP="plugdev"\n'
        '       SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"\n'
        "     then reload udev, add yourself to `plugdev`, and replug the device.\n"
        "  3. Install LiteRT: `pip install ai-edge-litert` (current Python; do\n"
        "     not use the deprecated `tflite-runtime` or the stale `pycoral`\n"
        "     wheels, which have no releases for recent Pythons).\n"
        "  4. Compile the model first: fully-quantized `.tflite` + "
        "`edgetpu_compiler` (see `onnxsim.export_edgetpu`).\n"
        "Without a device, the same LiteRT interpreter runs the uncompiled\n"
        "quantized model on CPU by omitting the delegate."
    )


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------

# TFLite ops the Edge TPU executes, per
# https://coral.ai/docs/edgetpu/models-intro/#supported-operations
# (compiler 16.0 / runtime 14). Anything else in a quantized model falls back
# to the CPU (the compiler partitions the graph at the first unsupported op).
EDGETPU_SUPPORTED_TFLITE_OPS = frozenset(
    {
        "ADD",
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "EXPAND_DIMS",
        "FULLY_CONNECTED",
        "L2_NORMALIZATION",
        "LOGISTIC",
        "LSTM",
        "MAXIMUM",
        "MAX_POOL_2D",
        "MEAN",
        "MINIMUM",
        "MUL",
        "PACK",
        "PAD",
        "PRELU",
        "QUANTIZE",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "RELU",
        "RELU6",
        "RELU_N1_TO_1",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "RSQRT",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "SUM",
        "SQUARED_DIFFERENCE",
        "TANH",
        "TRANSPOSE",
        "TRANSPOSE_CONV",
    }
)

# TFLite ops the compiler leaves at the model boundaries / outside the TPU
# subgraph in specific situations rather than mapping: informational, not
# failures on their own.
_EDGETPU_BOUNDARY_TFLITE_OPS = frozenset({"QUANTIZE", "DEQUANTIZE"})

# ONNX op -> (expected TFLite op(s), status, note). Status is one of
# "supported", "conditional" (maps only within the stated limits), or
# "unsupported" (runs on CPU; the compiler partitions the graph there).
# Statuses marked "(converter)" depend on a TF-converter fusion rather than on
# the translator's own lowering; they were verified against edgetpu_compiler
# 16.0 where noted.
_ONNX_EDGETPU_SUPPORT: Dict[str, Tuple[str, str, str]] = {
    "Add": ("ADD", "supported", ""),
    "Sub": ("SUB", "supported", ""),
    "Mul": ("MUL", "supported", ""),
    "Div": (
        "DIV",
        "unsupported",
        "no integer DIV kernel; dividing by a constant is usually fused into MUL by the converter, other divisions run on CPU",
    ),
    "Relu": ("RELU", "supported", ""),
    "Clip": (
        "MAXIMUM/MINIMUM",
        "supported",
        "constant bounds fold into clamp ops (converter)",
    ),
    "Sigmoid": ("LOGISTIC", "supported", ""),
    "Tanh": ("TANH", "supported", ""),
    "PRelu": (
        "RELU/MINIMUM/MUL/ADD",
        "supported",
        "onnxsim lowers PRelu to Relu/Minimum/Mul/Add so every emitted op maps",
    ),
    "LeakyRelu": (
        "LEAKY_RELU",
        "unsupported",
        "TFLite has no integer LEAKY_RELU kernel; use Relu for Edge TPU models",
    ),
    "Gelu": (
        "GELU",
        "unsupported",
        "unknown to edgetpu_compiler 16.0 entirely (builtin out of range); runs on CPU",
    ),
    "Erf": (
        "(no kernel)",
        "unsupported",
        "no Edge TPU kernel; implies Gelu is also unsupported",
    ),
    "Softmax": (
        "SOFTMAX",
        "conditional",
        "only 1-D inputs with at most 16000 elements map fully",
    ),
    "Conv": (
        "CONV_2D/DEPTHWISE_CONV_2D",
        "conditional",
        "2-D only; depthwise needs group == in-channels; equal x/y dilation",
    ),
    "Gemm": ("FULLY_CONNECTED", "supported", ""),
    "MatMul": ("FULLY_CONNECTED", "supported", ""),
    "MaxPool": (
        "MAX_POOL_2D",
        "conditional",
        "no fused activation, no dilation, no ceil_mode",
    ),
    "AveragePool": (
        "AVERAGE_POOL_2D",
        "conditional",
        "no fused activation; onnxsim tiles the count_include_pad=0 divisor so padded pooling still compiles",
    ),
    "GlobalAveragePool": ("MEAN", "supported", ""),
    "GlobalMaxPool": ("REDUCE_MAX", "supported", "needs runtime >= 14"),
    "ReduceMean": ("MEAN", "conditional", "no batch-dimension reduction"),
    "ReduceSum": (
        "SUM",
        "conditional",
        "no batch-dimension reduction; needs runtime >= 13",
    ),
    "ReduceMax": (
        "REDUCE_MAX",
        "conditional",
        "no batch-dimension reduction; needs runtime >= 14",
    ),
    "ReduceMin": (
        "REDUCE_MIN",
        "conditional",
        "no batch-dimension reduction; needs runtime >= 14",
    ),
    "ReduceProd": ("(no kernel)", "unsupported", "no Edge TPU kernel"),
    "BatchNormalization": (
        "MUL/SUB/ADD",
        "supported",
        "constant scale/bias/mean/var fold into multiply-adds (converter)",
    ),
    "Reshape": ("RESHAPE", "supported", "some very large reshapes may not map"),
    "Flatten": ("RESHAPE", "supported", ""),
    "Squeeze": ("SQUEEZE", "supported", ""),
    "Unsqueeze": ("EXPAND_DIMS", "supported", "needs runtime >= 13"),
    "Transpose": ("TRANSPOSE", "supported", "needs runtime >= 14"),
    "Concat": (
        "CONCATENATION",
        "conditional",
        "no fused activation; a constant input forces exactly 2 inputs and all-zero constants",
    ),
    "Split": ("SPLIT", "conditional", "no batch-dimension split"),
    "Gather": ("GATHER", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Tile": ("TILE", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Pad": ("PAD", "conditional", "no batch-dimension padding"),
    "Slice": (
        "SLICE/STRIDED_SLICE",
        "conditional",
        "strided slices need all strides == 1",
    ),
    "Shape": (
        "SHAPE",
        "unsupported",
        "shape queries run on CPU; simplify the model so they constant-fold away",
    ),
    "Constant": (
        "(folded)",
        "supported",
        "constants fold into the graph during conversion",
    ),
    "Cast": (
        "CAST",
        "unsupported",
        "quantized graphs should not need casts; runs on CPU",
    ),
    "Identity": ("(removed)", "supported", "removed by the converter"),
    "Dropout": (
        "(removed)",
        "supported",
        "inference-mode Dropout is an identity and is removed",
    ),
    "Neg": ("NEG", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Abs": ("ABS", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Sqrt": (
        "SQRT",
        "unsupported",
        "no Edge TPU kernel (RSQRT needs runtime >= 14); runs on CPU",
    ),
    "Exp": ("EXP", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Log": ("LOG", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Pow": ("POW", "unsupported", "no Edge TPU kernel; runs on CPU"),
    "Max": ("MAXIMUM", "supported", ""),
    "Min": ("MINIMUM", "supported", ""),
}


@dataclass
class EdgeTPUNodeFinding:
    """One compatibility finding for a single ONNX node (or the graph itself)."""

    node: str
    op_type: str
    level: str  # "error" | "warning" | "info"
    message: str


@dataclass
class EdgeTPUCompatibilityReport:
    """Result of :func:`check_onnx_for_edgetpu`."""

    findings: List[EdgeTPUNodeFinding] = field(default_factory=list)

    @property
    def errors(self) -> List[EdgeTPUNodeFinding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> List[EdgeTPUNodeFinding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def fully_supported(self) -> bool:
        """True when nothing errors and no node is expected to fall back to CPU."""
        return not self.errors and not self.warnings

    def summary(self) -> str:
        lines = [
            f"Edge TPU compatibility: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s) in {len(self.findings)} finding(s)"
        ]
        for f in self.findings:
            lines.append(f"  [{f.level}] {f.node} ({f.op_type}): {f.message}")
        return "\n".join(lines)


def _static_dims(value_info: onnx.ValueInfoProto) -> Optional[List[int]]:
    t = value_info.type.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            return None
    return dims


# Activation-size envelope above which the default NCHW export risks the Edge
# TPU compiler's "large activation tensors" failure: measured with
# edgetpu_compiler 16.0 on 3x3 conv graphs, a 4-D activation with at least
# this many channels and this many elements fails to compile with the NCHW
# entry transpose (64ch x 32x32 fails; 64ch x 24x24, 32ch x 32x32 and 4ch x
# 128x128 map), while the identical NHWC graph maps fully. Conservative on
# purpose -- exit-transpose-only graphs are unaffected, which static analysis
# cannot tell apart.
_EDGETPU_LARGE_ACTIVATION_CHANNELS = 8
_EDGETPU_LARGE_ACTIVATION_ELEMENTS = 65536


def _largest_4d_activation(model: onnx.ModelProto) -> Optional[Tuple[int, int, str]]:
    """Largest 4-D activation as ``(channels, elements, tensor_name)``.

    Scans graph inputs/outputs plus shape-inferred intermediates (initializers
    excluded -- constant weights never transpose at runtime). Returns ``None``
    when no fully-static 4-D tensor is found or shape inference is unavailable.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    initializer_names = {t.name for t in model.graph.initializer}
    best: Optional[Tuple[int, int, str]] = None
    seen = set()
    candidates = (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    )
    for value_info in candidates:
        if value_info.name in seen or value_info.name in initializer_names:
            continue
        seen.add(value_info.name)
        dims = _static_dims(value_info)
        if dims is None or len(dims) != 4 or any(d <= 0 for d in dims):
            continue
        elements = int(np.prod(dims, dtype=np.int64))
        if elements == 0:
            continue
        if best is None or elements > best[1]:
            best = (dims[1], elements, value_info.name)
    return best


def check_onnx_for_edgetpu(model: onnx.ModelProto) -> EdgeTPUCompatibilityReport:
    """Statically check an ONNX model against the Edge TPU model requirements.

    Covers fully-static input shapes, the >3-D size rule (only the 3 innermost
    dimensions may exceed 1), the activation-size envelope that risks the
    compiler's "large activation tensors" failure under the default NCHW
    layout, and a per-node lookup in the Edge TPU operation table. This is a
    heuristic pre-check -- the authoritative answer for a converted model comes
    from :func:`compile_for_edgetpu`'s operator log -- but it needs nothing but
    ``onnx`` and runs before any conversion.

    Parameters
    ----------
    model:
        The ONNX model to check (usually the output of :func:`onnxsim.simplify`).

    Returns
    -------
    EdgeTPUCompatibilityReport
    """
    report = EdgeTPUCompatibilityReport()
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}

    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        if _static_dims(inp) is None:
            report.findings.append(
                EdgeTPUNodeFinding(
                    node=inp.name,
                    op_type="graph input",
                    level="error",
                    message="dynamic or missing dimensions; the Edge TPU needs "
                    "fully static shapes (pin them with onnxsim's "
                    "--overwrite-input-shape/--test-input-shape).",
                )
            )

    for value_info in list(graph.input) + list(graph.output):
        dims = _static_dims(value_info)
        if dims is not None and len(dims) > 3 and any(d != 1 for d in dims[:-3]):
            report.findings.append(
                EdgeTPUNodeFinding(
                    node=value_info.name,
                    op_type="tensor shape",
                    level="error",
                    message=f"shape {dims} has more than 3 dimensions with a "
                    "leading dimension > 1; only the 3 innermost dimensions "
                    "may exceed 1 on the Edge TPU.",
                )
            )

    largest = _largest_4d_activation(model)
    if largest is not None:
        channels, elements, tensor_name = largest
        if (
            channels >= _EDGETPU_LARGE_ACTIVATION_CHANNELS
            and elements >= _EDGETPU_LARGE_ACTIVATION_ELEMENTS
        ):
            report.findings.append(
                EdgeTPUNodeFinding(
                    node=tensor_name,
                    op_type="activation size",
                    level="warning",
                    message=f"4-D activation with {channels} channels and "
                    f"{elements} elements; the default NCHW export risks the "
                    "Edge TPU compiler's 'large activation tensors' failure "
                    "on the entry transpose at this size (measured: 64ch x "
                    "32x32 fails, the identical NHWC graph maps fully). "
                    "Convert with io_layout='nhwc' (--tflite-layout nhwc) for "
                    "channel-last I/O with no transposes, or confirm with "
                    "compile_for_edgetpu.",
                )
            )

    for node in graph.node:
        entry = _ONNX_EDGETPU_SUPPORT.get(node.op_type)
        name = node.name or (node.output[0] if node.output else node.op_type)
        if entry is None:
            report.findings.append(
                EdgeTPUNodeFinding(
                    node=name,
                    op_type=node.op_type,
                    level="warning",
                    message="op is not in onnxsim's Edge TPU table (and not in "
                    "the builtin TFLite translator); it will run on CPU, if "
                    "it converts at all.",
                )
            )
            continue
        _, status, note = entry
        if status == "unsupported":
            report.findings.append(
                EdgeTPUNodeFinding(
                    node=name,
                    op_type=node.op_type,
                    level="warning",
                    message=f"expected to fall back to CPU ({entry[0]}). {note}".rstrip(),
                )
            )
        elif status == "conditional":
            report.findings.append(
                EdgeTPUNodeFinding(
                    node=name,
                    op_type=node.op_type,
                    level="info",
                    message=f"maps only within limits ({entry[0]}). {note}".rstrip(),
                )
            )
    return report


@dataclass
class EdgeTPUTfliteOpFinding:
    """One TFLite operator's compilation status from the operator log."""

    op: str
    count: int
    status: str

    @property
    def mapped(self) -> bool:
        return "Mapped to Edge TPU" in self.status


@dataclass
class TfliteCompatibilityReport:
    """Result of :func:`check_tflite_for_edgetpu`."""

    operators: List[EdgeTPUTfliteOpFinding] = field(default_factory=list)
    float_io_boundaries: bool = False

    @property
    def unmapped(self) -> List[EdgeTPUTfliteOpFinding]:
        return [o for o in self.operators if not o.mapped]

    @property
    def fully_supported(self) -> bool:
        return not self.unmapped

    def summary(self) -> str:
        lines = [
            f"TFLite Edge TPU compatibility: "
            f"{sum(o.count for o in self.operators) - sum(o.count for o in self.unmapped)}/"
            f"{sum(o.count for o in self.operators)} ops supported"
        ]
        for o in self.operators:
            lines.append(f"  {o.op:28s} x{o.count:<4d} {o.status}")
        if self.float_io_boundaries:
            lines.append(
                "  note: the model still uses float I/O (a CPU-side "
                "quantize/dequantize pair runs at each boundary); re-convert with "
                'inference_io_dtype="uint8"/"int8" for fully-quantized I/O.'
            )
        return "\n".join(lines)


def check_tflite_for_edgetpu(tflite_model: bytes) -> TfliteCompatibilityReport:
    """Check a ``.tflite`` flatbuffer against the Edge TPU operation table.

    Parses the model's operator codes with LiteRT's own flatbuffer schema
    (``ai_edge_litert`` must be installed) and reports which ops the Edge TPU
    executes and which would fall back to the CPU -- without needing the
    compiler binary.

    Parameters
    ----------
    tflite_model:
        The serialized ``.tflite`` flatbuffer.

    Returns
    -------
    TfliteCompatibilityReport
    """
    try:
        from ai_edge_litert.tools import flatbuffer_utils as fbu
    except ImportError as exc:
        raise RuntimeError(
            "Checking a .tflite model needs LiteRT's flatbuffer schema. "
            + _LITERT_INSTALL_HINT
        ) from exc

    model = fbu.convert_bytearray_to_object(bytearray(tflite_model))
    builtin_names = {
        v: k for k, v in vars(fbu.BuiltinOperator).items() if isinstance(v, int)
    }
    counts: Dict[str, int] = {}
    float_io = False
    for subgraph in model.subgraphs:
        for i in list(subgraph.inputs) + list(subgraph.outputs):
            if subgraph.tensors[i].type == fbu.TensorType.FLOAT32:
                float_io = True
        for op in subgraph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            builtin = int(fbu.get_builtin_code_from_operator_code(code))
            name = builtin_names.get(builtin)
            if name is None:
                # Custom op (e.g. the Edge TPU custom op in an already-compiled
                # model, or a Flex op): never maps, always worth reporting.
                custom = getattr(code, "customCode", None)
                if isinstance(custom, (bytes, bytearray)):
                    custom = bytes(custom).decode("utf-8", "replace")
                name = f"CUSTOM({custom})" if custom else f"UNKNOWN({builtin})"
            counts[name] = counts.get(name, 0) + 1

    operators = []
    for name in sorted(counts):
        if name in EDGETPU_SUPPORTED_TFLITE_OPS:
            status = "Mapped to Edge TPU"
        elif name in _EDGETPU_BOUNDARY_TFLITE_OPS:
            status = (
                "Boundary quantize op: maps only with quantized I/O "
                "(see inference_io_dtype=)"
            )
        else:
            status = "Operation not supported: runs on CPU"
        operators.append(EdgeTPUTfliteOpFinding(name, counts[name], status))
    return TfliteCompatibilityReport(operators, float_io)


# ---------------------------------------------------------------------------
# Full-integer quantization
# ---------------------------------------------------------------------------


def quantize_for_edgetpu(
    model: onnx.ModelProto,
    representative_dataset: Any = None,
    num_calibration_samples: int = 100,
    inference_io_dtype: Any = "uint8",
    seed: int = 0,
    io_layout: str = "nchw",
    **kwargs: Any,
) -> bytes:
    """Convert an ONNX model to a fully-integer-quantized ``.tflite`` model.

    This is :func:`onnxsim.tflite_export.convert_to_tflite` with
    ``int8_quantize=True`` and Edge TPU-suitable defaults (quantized I/O):
    the quantization the Edge TPU requires before ``edgetpu_compiler`` can
    map the graph. Without ``representative_dataset``, calibration data is
    uniform-random (see
    :func:`onnxsim.tflite_export.random_representative_dataset`); pass real,
    representative inputs for production accuracy.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    representative_dataset:
        Calibration batches callable (TensorFlow representative-dataset
        protocol), or ``None`` for random data. With ``io_layout="nhwc"`` the
        batches must carry 4-D inputs in channel-last order.
    num_calibration_samples:
        Random batches to generate when ``representative_dataset`` is ``None``.
    inference_io_dtype:
        ``"uint8"`` (default) or ``"int8"``: the quantized model I/O type.
    seed:
        Seed for the random calibration data.
    io_layout:
        ``"nchw"`` (default) keeps public tensors in ONNX order;
        ``"nhwc"`` carries 4-D tensors channel-last end to end, emitting no
        transposes -- required for larger models, whose NCHW entry transpose
        the Edge TPU compiler refuses (see ``io_layout`` in
        :func:`onnxsim.tflite_export.convert_to_tflite`).
    **kwargs:
        Forwarded to :func:`onnxsim.tflite_export.convert_to_tflite`
        (e.g. ``backend="builtin"``).

    Returns
    -------
    bytes
        The serialized quantized ``.tflite`` flatbuffer (not yet compiled --
        pass it to :func:`compile_for_edgetpu`).
    """
    from onnxsim import tflite_export

    if representative_dataset is None:
        representative_dataset = tflite_export.random_representative_dataset(
            model, num_calibration_samples, seed=seed, io_layout=io_layout
        )
    return tflite_export.convert_to_tflite(
        model,
        int8_quantize=True,
        representative_dataset=representative_dataset,
        inference_io_dtype=inference_io_dtype,
        io_layout=io_layout,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


@dataclass
class EdgeTPUCompileResult:
    """Result of :func:`compile_for_edgetpu`."""

    success: bool
    operators: List[EdgeTPUTfliteOpFinding] = field(default_factory=list)
    num_subgraphs: int = 0
    compiled_model: Optional[bytes] = None
    output_path: Optional[str] = None
    log_text: str = ""

    @property
    def mapped_ops(self) -> int:
        return sum(o.count for o in self.operators if o.mapped)

    @property
    def total_ops(self) -> int:
        return sum(o.count for o in self.operators)

    @property
    def fully_mapped(self) -> bool:
        return self.success and self.total_ops > 0 and self.mapped_ops == self.total_ops

    def summary(self) -> str:
        lines = [
            f"Edge TPU compilation: {'succeeded' if self.success else 'FAILED'}; "
            f"{self.mapped_ops}/{self.total_ops} ops mapped to the Edge TPU "
            f"({self.num_subgraphs} subgraph(s))"
        ]
        for o in self.operators:
            lines.append(f"  {o.op:28s} x{o.count:<4d} {o.status}")
        return "\n".join(lines)


def _parse_compiler_output(text: str) -> Tuple[List[EdgeTPUTfliteOpFinding], int]:
    operators: List[EdgeTPUTfliteOpFinding] = []
    num_subgraphs = 0
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Number of Edge TPU subgraphs:"):
            try:
                num_subgraphs = int(stripped.split(":")[1].strip().split()[0])
            except ValueError:
                pass
        if stripped.startswith("Operator") and "Status" in stripped:
            in_table = True
            continue
        if in_table:
            if stripped.startswith("Compilation"):
                break
            if not stripped:
                continue
            # Rows look like: "CONV_2D   1   Mapped to Edge TPU".
            # The op name is one token (custom-op rows aside); split off the
            # trailing status by locating the count column.
            parts = stripped.split()
            if len(parts) >= 3 and parts[1].isdigit():
                operators.append(
                    EdgeTPUTfliteOpFinding(parts[0], int(parts[1]), " ".join(parts[2:]))
                )
    return operators, num_subgraphs


def compile_for_edgetpu(
    tflite_model: bytes,
    output_path: Optional[str] = None,
    compiler: Optional[str] = None,
    out_dir: Optional[str] = None,
    extra_args: Sequence[str] = (),
    timeout: int = 600,
) -> EdgeTPUCompileResult:
    """Compile a fully-quantized ``.tflite`` model for the Edge TPU.

    Runs ``edgetpu_compiler`` on ``tflite_model`` (which must use
    full-integer quantization -- see :func:`quantize_for_edgetpu`) and parses
    its operator log into per-op TPU/CPU statuses. A model with unsupported
    ops still "succeeds" with those ops partitioned to the CPU; only a
    compiler failure (non-zero exit, e.g. unbroadcastable operands from an
    exotic lowering) reports ``success=False``.

    Parameters
    ----------
    tflite_model:
        The serialized quantized ``.tflite`` flatbuffer.
    output_path:
        Where to write the compiled ``*_edgetpu.tflite`` model. Defaults to a
        temporary directory (the bytes are still returned in
        :attr:`EdgeTPUCompileResult.compiled_model`); when ``out_dir`` is
        given without ``output_path``, the compiler's default
        ``<stem>_edgetpu.tflite`` name inside ``out_dir`` is used.
    compiler:
        Explicit path to the ``edgetpu_compiler`` binary (else
        ``$EDGETPU_COMPILER`` / ``PATH`` is searched).
    out_dir:
        Directory for the compiler's outputs (model + ``.log``).
    extra_args:
        Extra ``edgetpu_compiler`` flags, e.g. ``["--min_runtime_version", "14"]``.
    timeout:
        Compiler timeout in seconds.

    Returns
    -------
    EdgeTPUCompileResult
    """
    compiler_path = find_edgetpu_compiler(compiler)
    if compiler_path is None:
        raise RuntimeError(_EDGETPU_COMPILER_HINT)

    with tempfile.TemporaryDirectory(prefix="onnxsim_edgetpu_") as tmp_dir:
        src = os.path.join(tmp_dir, "model.tflite")
        with open(src, "wb") as f:
            f.write(tflite_model)
        out = out_dir or tmp_dir
        os.makedirs(out, exist_ok=True)
        # -s prints the operator mapping table to stdout (without it the
        # table only lands in the .log file); the log file is still read as a
        # fallback below.
        cmd = [compiler_path, "-s", *extra_args, "-o", out, src]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"Failed to run the Edge TPU compiler at {compiler_path!r}: {exc}"
            ) from exc
        text = (proc.stdout or "") + (proc.stderr or "")
        operators, num_subgraphs = _parse_compiler_output(text)
        success = proc.returncode == 0 and (
            "compiled successfully" in text or "Compilation succeeded" in text
        )
        compiled_bytes: Optional[bytes] = None
        compiled_path: Optional[str] = None
        if success:
            # The compiler writes <stem>_edgetpu.tflite next to the input stem.
            stem = os.path.splitext(os.path.basename(src))[0]
            default_name = os.path.join(out, stem + "_edgetpu.tflite")
            if not operators:
                # No table on stdout (e.g. the caller overrode -s via
                # extra_args): fall back to the .log file.
                log_path = os.path.join(out, stem + "_edgetpu.log")
                if os.path.isfile(log_path):
                    with open(log_path) as f:
                        log_text = f.read()
                    text += "\n" + log_text
                    operators, num_subgraphs = _parse_compiler_output(text)
            if os.path.isfile(default_name):
                with open(default_name, "rb") as f:
                    compiled_bytes = f.read()
                if output_path is not None:
                    with open(output_path, "wb") as f:
                        f.write(compiled_bytes)
                    compiled_path = output_path
                else:
                    compiled_path = default_name if out_dir else None
        return EdgeTPUCompileResult(
            success=success,
            operators=operators,
            num_subgraphs=num_subgraphs,
            compiled_model=compiled_bytes,
            output_path=compiled_path,
            log_text=text,
        )


@dataclass
class EdgeTPUExportResult:
    """Result of :func:`export_edgetpu`: the quantized and compiled models."""

    quantized_model: bytes
    compile_result: EdgeTPUCompileResult


def export_edgetpu(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    representative_dataset: Any = None,
    num_calibration_samples: int = 100,
    inference_io_dtype: Any = "uint8",
    compiler: Optional[str] = None,
    extra_compiler_args: Sequence[str] = (),
    io_layout: str = "nchw",
    **kwargs: Any,
) -> EdgeTPUExportResult:
    """Convert an ONNX model to an Edge TPU-compiled ``.tflite`` model.

    One-shot chaining of :func:`quantize_for_edgetpu` and
    :func:`compile_for_edgetpu`: full-integer quantization (random calibration
    data unless ``representative_dataset`` is given) followed by
    ``edgetpu_compiler``.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    output_path:
        If given, the compiled ``*_edgetpu.tflite`` model is written here.
    representative_dataset / num_calibration_samples / inference_io_dtype:
        See :func:`quantize_for_edgetpu`.
    compiler / extra_compiler_args:
        See :func:`compile_for_edgetpu`.
    io_layout:
        See :func:`quantize_for_edgetpu`; ``"nhwc"`` is recommended for larger
        models (see :func:`check_onnx_for_edgetpu`'s activation-size warning).
    **kwargs:
        Forwarded to :func:`onnxsim.tflite_export.convert_to_tflite`.

    Returns
    -------
    EdgeTPUExportResult
    """
    quantized = quantize_for_edgetpu(
        model,
        representative_dataset=representative_dataset,
        num_calibration_samples=num_calibration_samples,
        inference_io_dtype=inference_io_dtype,
        io_layout=io_layout,
        **kwargs,
    )
    compiled = compile_for_edgetpu(
        quantized,
        output_path=output_path,
        compiler=compiler,
        extra_args=extra_compiler_args,
    )
    if not compiled.success:
        raise RuntimeError(
            "Edge TPU compilation failed:\n" + compiled.log_text.strip()[-4000:]
        )
    return EdgeTPUExportResult(quantized, compiled)


# ---------------------------------------------------------------------------
# LiteRT inference
# ---------------------------------------------------------------------------


def _interpreter_backend():
    """Return the (module, kind) to build interpreters with: LiteRT first,
    then TensorFlow's own Lite runtime, then tflite_runtime."""
    try:
        from ai_edge_litert import interpreter as litert

        return litert, "litert"
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.lite, "tensorflow"
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tf_lite_runtime

        return tf_lite_runtime, "tflite_runtime"
    except ImportError:
        pass
    raise RuntimeError("No TFLite runtime is installed. " + _LITERT_INSTALL_HINT)


def make_litert_interpreter(
    model: Any,
    use_edgetpu: bool = False,
    num_threads: Optional[int] = None,
    edgetpu_library: Optional[str] = None,
    prefer_litert: bool = True,
):
    """Build a TFLite interpreter, preferring LiteRT.

    Parameters
    ----------
    model:
        Either the serialized ``.tflite`` flatbuffer (``bytes``) or a path to
        a ``.tflite`` file.
    use_edgetpu:
        Load the Edge TPU delegate (``libedgetpu``) so compiled subgraphs run
        on the device. Requires the Edge TPU runtime plus a plugged-in device;
        failures raise a ``RuntimeError`` ending with :func:`edgetpu_setup_hint`.
    num_threads:
        CPU threads for CPU kernels.
    edgetpu_library:
        Explicit path to ``libedgetpu.so.1`` (else ``$EDGETPU_LIBRARY`` / the
        system library paths are searched).
    prefer_litert:
        Use ``ai_edge_litert`` when installed (default), falling back to
        ``tf.lite`` and then ``tflite_runtime``. Set to ``False`` to skip
        LiteRT even when it is installed.

    Returns
    -------
    An interpreter with the usual ``allocate_tensors`` / ``set_tensor`` /
    ``invoke`` / ``get_tensor`` API.
    """
    backend = None
    if prefer_litert:
        try:
            from ai_edge_litert import interpreter as litert

            backend = litert
        except ImportError:
            backend = None
    if backend is None:
        backend, _ = _interpreter_backend()
    kwargs: Dict[str, Any] = {}
    if isinstance(model, (bytes, bytearray)):
        kwargs["model_content"] = bytes(model)
    else:
        kwargs["model_path"] = model
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    if use_edgetpu:
        library = find_edgetpu_library(edgetpu_library)
        if library is None:
            raise RuntimeError(
                "use_edgetpu=True but libedgetpu could not be found.\n"
                + edgetpu_setup_hint()
            )
        try:
            delegate = backend.load_delegate(library)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the Edge TPU delegate from {library!r}: {exc}. "
                "The usual causes are a missing/busy device or USB nodes the "
                "current user cannot write to.\n" + edgetpu_setup_hint()
            ) from exc
        kwargs["experimental_delegates"] = [delegate]
    return backend.Interpreter(**kwargs)


def _match_inputs(
    interpreter: Any, inputs: Mapping[str, np.ndarray]
) -> List[Tuple[int, np.ndarray]]:
    details = interpreter.get_input_details()
    if len(details) == 1 and len(inputs) == 1:
        (detail,) = details
        (value,) = inputs.values()
        return [(detail["index"], np.asarray(value, dtype=detail["dtype"]))]
    available = {d["name"]: d for d in details}
    matched = []
    for name, value in inputs.items():
        detail = available.get(name)
        if detail is None:
            # TFLite tensor names carry tracing suffixes (e.g. "x:0");
            # also accept the suffix-stripped name.
            bare = {n.split(":")[0].split("/")[-1]: d for n, d in available.items()}
            detail = bare.get(name.split(":")[0].split("/")[-1])
        if detail is None:
            raise KeyError(
                f"Unknown model input {name!r}; available inputs: {sorted(available)}"
            )
        matched.append((detail["index"], np.asarray(value, dtype=detail["dtype"])))
    return matched


def run_litert(
    model: Any,
    inputs: Mapping[str, np.ndarray],
    use_edgetpu: bool = False,
    **kwargs: Any,
) -> List[np.ndarray]:
    """Run a ``.tflite`` model with LiteRT and return the output arrays.

    Parameters
    ----------
    model:
        Serialized ``.tflite`` flatbuffer (``bytes``) or a path to one.
    inputs:
        Model inputs by tensor name (a single input may be given under any key).
    use_edgetpu:
        Run compiled subgraphs on the Edge TPU via the ``libedgetpu`` delegate.
    **kwargs:
        Forwarded to :func:`make_litert_interpreter`.

    Returns
    -------
    list[numpy.ndarray]
        One array per model output, in output order.
    """
    interpreter = make_litert_interpreter(model, use_edgetpu=use_edgetpu, **kwargs)
    interpreter.allocate_tensors()
    for index, value in _match_inputs(interpreter, inputs):
        interpreter.set_tensor(index, value)
    interpreter.invoke()
    return [
        interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()
    ]
