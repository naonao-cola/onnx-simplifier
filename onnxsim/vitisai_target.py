"""Target legalizer and support checker for AMD's Ryzen AI NPU (VitisAI EP).

The Vitis AI execution provider partitions the graph into NPU/CPU subgraphs
transparently, but its compiler front end is strict about two things this
module covers -- both measured, not guessed, on a Strix Halo NPU with Ryzen
AI Software 1.8 (Vitis AI EP build of ONNX Runtime 1.27.0, XRT 2.25):

- **``Conv`` with default attributes aborts the process.** A ``Conv`` that
  relies on ONNX defaults (no ``strides``/``pads``/``dilations``/
  ``kernel_shape``/``group`` attributes -- exactly what
  ``onnxruntime.quantization.quantize_static`` and AMD Quark's ``XINT8``/
  ``A8W8`` recipes emit) dies inside XIR conversion with ``SIGABRT``::

      Check failed: ... xir::Op{type = conv2d} :
      Attr stride has type REQUIRED, but not set.

  :func:`legalize_for_vitisai` materializes those five attributes explicitly
  (resolving the weight through ``DequantizeLinear`` chains, so QDQ models
  are covered), which turns the abort into a normal session -- and, for the
  quantized ``Conv+Relu``/``MatMul+Add`` shapes the EP fuses, into an NPU
  offload (verified via the EP's own ``DPU subgraph`` log lines, with
  bit-exact NPU-vs-CPU numerics on the quantized conv probe).
- **Some shapes never reach the NPU at all.** :func:`check_vitisai_support`
  flags the ones with a hard failure attached: the default-attribute
  ``Conv`` above, ``LSTM`` nodes (measured segfault during session
  creation), and bf16-typed tensors (rejected with ``INVALID_GRAPH`` --
  BF16 *execution* on this EP means an fp32 graph plus ``config_file``,
  not a bf16-typed graph). Plain CPU fallback (standalone ``Gelu``/
  ``Softmax``/``LayerNorm``/data-movement ops, float-bias ``MatMul+Add``)
  is by design and is not flagged: the EP handles it gracefully.

Meant to be called on a model about to be handed to a ``VitisAIExecutionProvider``
session -- typically after :func:`onnxsim.simplify` (which does *not*
materialize default attributes itself, so legalizing afterwards is still
needed) and, for INT8 deployment, after quantization (Quark ``XINT8`` is the
recommended recipe -- see the README's AMD NPU section).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Union

import onnx

#: ``Conv`` attributes the EP's XIR lowering requires to be present. A ``Conv``
#: missing any of these aborts session creation (see this module's docstring)
#: instead of falling back to the CPU.
_REQUIRED_CONV_ATTRS = ("strides", "pads", "dilations", "kernel_shape", "group")


def _load(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    if isinstance(model, str):
        return onnx.load(model, load_external_data=False)
    return model


def _weight_dims(graph: onnx.GraphProto, value_name: str) -> Optional[List[int]]:
    """Static dims of the weight behind ``value_name``, following it through
    ``DequantizeLinear`` producers (the QDQ shape every int8 pipeline emits),
    or ``None`` when the weight isn't a static initializer (dynamic weight --
    nothing to materialize from)."""
    inits = {t.name: t for t in graph.initializer}
    producers = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node
    name = value_name
    while True:
        init = inits.get(name)
        if init is not None:
            return list(init.dims)
        producer = producers.get(name)
        if producer is None or producer.op_type != "DequantizeLinear":
            return None
        name = producer.input[0]


def _materialize_conv_defaults(graph: onnx.GraphProto) -> List[str]:
    """Fill unset ``Conv`` attributes explicitly, in place. Returns one
    human-readable message per rewritten node (empty when nothing matched)."""
    messages = []
    for node in graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        missing = [
            attr
            for attr in _REQUIRED_CONV_ATTRS
            if not any(a.name == attr for a in node.attribute)
        ]
        if not missing:
            continue
        dims = _weight_dims(graph, node.input[1])
        if dims is None or len(dims) < 3:
            # Dynamic (non-initializer) weight, or a rank that isn't a valid
            # convolution weight at all -- leave it for shape inference and
            # validation to judge rather than guessing.
            continue
        spatial = len(dims) - 2
        defaults: Dict[str, Sequence[int]] = {
            "strides": [1] * spatial,
            "pads": [0] * (2 * spatial),
            "dilations": [1] * spatial,
            "kernel_shape": list(dims[2:]),
            "group": [1],
        }
        label = node.name or (node.output[0] if node.output else "<unnamed>")
        filled = []
        for attr in _REQUIRED_CONV_ATTRS:
            if attr in missing:
                value = defaults[attr]
                node.attribute.append(
                    onnx.helper.make_attribute(
                        attr, value[0] if attr == "group" else list(value)
                    )
                )
                filled.append(attr)
        messages.append(
            f"Conv node {label!r}: materialized default attributes "
            f"({', '.join(filled)}) required by the Vitis AI EP."
        )
    return messages


def legalize_for_vitisai(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """Rewrite ``model`` so the Vitis AI EP accepts it, returning a new model.

    Currently a single rewrite -- materializing default ``Conv`` attributes
    (see this module's docstring for why the EP needs them) -- applied to a
    copy; ``model`` itself is never mutated. The result is still plain,
    checker-valid ONNX: every other runtime accepts it exactly as before,
    so legalizing unconditionally is safe even when the model never runs on
    the NPU.

    :param model: the onnx ModelProto to legalize, or a file path
    :returns: the legalized onnx ModelProto (a distinct object)
    """
    source = _load(model)
    result = copy.deepcopy(source)
    _materialize_conv_defaults(result.graph)
    return result


def _default_attr_conv_nodes(graph: onnx.GraphProto) -> List[str]:
    """Labels of ``Conv`` nodes missing any attribute in
    :data:`_REQUIRED_CONV_ATTRS` -- each one is a process-abort risk on the
    EP (see this module's docstring), not a graceful CPU fallback."""
    labels = []
    for node in graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        present = {a.name for a in node.attribute}
        if any(attr not in present for attr in _REQUIRED_CONV_ATTRS):
            labels.append(node.name or (node.output[0] if node.output else "<unnamed>"))
    return labels


def _lstm_node_labels(graph: onnx.GraphProto) -> List[str]:
    """Labels of ``LSTM`` nodes -- session creation with the Vitis AI EP was
    measured segfaulting on these (Strix Halo, Ryzen AI 1.8)."""
    return [
        node.name or (node.output[0] if node.output else "<unnamed>")
        for node in graph.node
        if node.op_type == "LSTM"
    ]


def _bf16_tensor_names(model: onnx.ModelProto) -> List[str]:
    """Names of bf16-typed graph inputs, initializers and node outputs. The
    EP rejects bf16-typed graphs with ``INVALID_GRAPH``; BF16 *execution*
    means an fp32 graph plus the EP's ``config_file`` option instead."""
    names = []
    for value_info in list(model.graph.input) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
            names.append(value_info.name)
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.BFLOAT16:
            names.append(init.name or "<unnamed initializer>")
    for node in model.graph.node:
        for attr in node.attribute:
            tensors = []
            if attr.HasField("t"):
                tensors.append(attr.t)
            tensors.extend(attr.tensors)
            for tensor in tensors:
                if tensor.data_type == onnx.TensorProto.BFLOAT16:
                    names.append(
                        node.name or (node.output[0] if node.output else "<unnamed>")
                    )
                    break
    return names


def check_vitisai_support(model: Union[str, onnx.ModelProto]) -> List[str]:
    """Scans for the Vitis AI EP gaps described in this module's docstring:
    default-attribute ``Conv`` nodes (process abort), ``LSTM`` nodes
    (measured segfault), and bf16-typed tensors (rejected graph).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending node/tensor (empty if
            none). This is advisory only -- it does not modify ``model`` or
            raise, since every flagged graph is still valid ONNX, just not
            NPU-safe. Pair it with :func:`legalize_for_vitisai`, which fixes
            the ``Conv`` case outright.
    """
    source = _load(model)
    messages = [
        f"Conv node {label!r} relies on default attributes; the Vitis AI EP "
        "aborts session creation on these (XIR conv2d stride REQUIRED). Run "
        "onnxsim.legalize_for_vitisai first."
        for label in _default_attr_conv_nodes(source.graph)
    ]
    messages += [
        f"LSTM node {label!r} segfaulted Vitis AI EP session creation in "
        "measurement (Ryzen AI 1.8); keep it on the CPU provider."
        for label in _lstm_node_labels(source.graph)
    ]
    messages += [
        f"Tensor {name!r} is bf16-typed; the Vitis AI EP rejects bf16-typed "
        "graphs (INVALID_GRAPH). Run BF16 execution as an fp32 graph with "
        "the EP's config_file option instead."
        for name in _bf16_tensor_names(source)
    ]
    return messages
