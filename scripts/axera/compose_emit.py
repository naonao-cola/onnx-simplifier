"""Index retargeting for a Gather that is one segment of a composed AX650 graph.

``memory_emit`` retargets standalone one-op Gather models. A training step is
one ``neu mode`` node whose MCode holds many ops, so this module measures the
same operation when the Gather feeds a chain of further ops (see
``docs/axera-compose.md``). The measured chains all start with
``Gather(x[1,1,4,16], idx[8], axis=3)`` and continue with a prefix of
``Reshape([1,1,8,4]) -> MatMul(w[4,6], live input) -> Transpose(0,1,3,2) ->
Add(b[1,1,6,8], live input)``.

What Pulsar2 7.0-lite does with such a chain:

* the eight indices are still the first eight little-endian uint32 words of
  ``npu_params``, with the rest of the table following (quantization scales
  and zero padding);
* the MCode is not a prefix of the standalone Gather's MCode: it is rewritten
  for every chain length, and rebuilding the *same* graph changes hundreds of
  MCode bytes, so a compiler-built reference cannot be checked byte for byte
  against a fixture the way ``memory_emit`` does;
* the table tail depends on the calibration data, because the Gather output's
  scale is calibrated from the elements the indices select.

The emitter therefore keeps the reference's MCode and table tail untouched,
checks their structure (graph signature, table size, segment layout, and the
scale-free zero padding), and rewrites only the index words. Inputs must stay
inside the calibration range, and the reference should have been calibrated on
data whose gathered range covers the whole input.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
from collections.abc import Sequence

import mcode
import onnx
from memory_emit import _initializer, _mcode, _output_dims, _param_words

_HERE = os.path.dirname(os.path.abspath(__file__))
_INPUT_SHAPE = (1, 1, 4, 16)
_COUNT = 8

# chain name -> (fixture, graph input names, output shape)
_CHAINS = {
    "gather_reshape": (
        "compose_gather_reshape.axmodel.gz",
        ("x",),
        (1, 1, 8, 4),
    ),
    "gather_reshape_matmul": (
        "compose_gather_reshape_matmul.axmodel.gz",
        ("x", "w"),
        (1, 1, 8, 6),
    ),
    "gather_reshape_matmul_transpose": (
        "compose_gather_reshape_matmul_transpose.axmodel.gz",
        ("x", "w"),
        (1, 1, 6, 8),
    ),
    "gather_reshape_matmul_transpose_add": (
        "compose_gather_reshape_matmul_transpose_add.axmodel.gz",
        ("x", "w", "b"),
        (1, 1, 6, 8),
    ),
}


def measured_chains() -> tuple[str, ...]:
    return tuple(_CHAINS)


def _load_fixture(chain: str) -> onnx.ModelProto:
    with gzip.open(os.path.join(_HERE, "fixtures", _CHAINS[chain][0]), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _segment_layout(model: onnx.ModelProto) -> list[tuple[int, int]]:
    header, segs = mcode.segments(_mcode(model))
    return [(header, 0)] + [(offset, length) for offset, length, _ in segs]


def _check_signature(model: onnx.ModelProto, chain: str) -> None:
    _, inputs, out_shape = _CHAINS[chain]
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("reference must contain exactly one compiled 'neu mode' node")
    node = model.graph.node[0]
    # The compiler orders the NPU node's inputs its own way (e.g. ['w', 'x'] for
    # graph inputs ['x', 'w']), so only the set is checked.
    if sorted(node.input) != sorted(inputs) or list(node.output) != ["y"]:
        raise ValueError(
            f"reference NPU node must map inputs {sorted(inputs)} to output 'y'"
        )
    if [v.name for v in model.graph.input] != list(inputs):
        raise ValueError(f"reference must have exactly the inputs {list(inputs)}")
    if [v.name for v in model.graph.output] != ["y"]:
        raise ValueError("reference must have exactly one output named 'y'")
    x = model.graph.input[0].type.tensor_type
    if x.elem_type != onnx.TensorProto.FLOAT or (
        tuple(d.dim_value for d in x.shape.dim) != _INPUT_SHAPE
    ):
        raise ValueError(f"reference input 'x' must be float32 {list(_INPUT_SHAPE)}")
    if tuple(_output_dims(model.graph.output[0], "y")) != out_shape:
        raise ValueError(f"reference output must have shape {list(out_shape)}")
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
    }
    if json.loads(attrs.get("outputs_info", b"{}")) != {"y": ["FP32", list(out_shape)]}:
        raise ValueError("outputs_info disagrees with the output shape")
    dynamic = _initializer(model, "npu_dyn_params")
    if dynamic.raw_data or dynamic.dims != [0]:
        raise ValueError("reference must have the empty npu_dyn_params initializer")


def emit_gather_in_graph(
    chain: str,
    output_path: str,
    *,
    indices: Sequence[int],
    reference_path: str | None = None,
) -> str:
    """Retarget the Gather indices of a measured composed chain.

    ``chain`` names one of :func:`measured_chains`. ``indices`` are eight
    integers in ``[0, 15]`` (duplicates and any order). With no
    ``reference_path`` the committed compiler-built fixture is the reference;
    a compiler-built reference of the same graph is also accepted, since its
    MCode differs from the fixture's only by build noise. Its structure must
    match the fixture: signature, table size, segment layout, scale words, and
    the padding. The MCode and every table word after the indices are kept.
    """
    if chain not in _CHAINS:
        raise ValueError(f"unmeasured chain {chain!r}; measured: {list(_CHAINS)}")
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise ValueError("indices must be a sequence of eight integers")
    target = tuple(indices)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in target):
        raise ValueError("indices must contain integers")
    if len(target) != _COUNT or any(not 0 <= v < _INPUT_SHAPE[-1] for v in target):
        raise ValueError(
            f"indices must be {_COUNT} integers in the range [0, {_INPUT_SHAPE[-1] - 1}]"
        )

    template = _load_fixture(chain)
    if reference_path is None:
        model = template
    else:
        model = onnx.load(reference_path, load_external_data=False)
    _check_signature(model, chain)

    words = _param_words(model)
    template_words = _param_words(template)
    if len(words) != len(template_words):
        raise ValueError("reference npu_params size does not match the fixture")
    if any(v >= _INPUT_SHAPE[-1] for v in words[:_COUNT]):
        raise ValueError("reference Gather index is out of bounds")
    padding = [i for i, v in enumerate(template_words) if i >= _COUNT and v == 0]
    if any(words[i] for i in padding):
        raise ValueError("reference npu_params padding must be zero")
    if len(_mcode(model)) != len(_mcode(template)):
        raise ValueError("reference MCode size does not match the fixture")
    if _segment_layout(model) != _segment_layout(template):
        raise ValueError("reference MCode segment layout does not match the fixture")

    table = _initializer(model, "npu_params")
    table.raw_data = struct.pack(f"<{len(words)}I", *target, *words[_COUNT:])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
