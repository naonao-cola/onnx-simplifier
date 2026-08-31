"""Tests for ``onnxsim.gguf_reconstruct``'s gpt-oss-20b MoE/FFN building
blocks: :func:`onnxsim.gguf_reconstruct._interleave_gate_up` and
:func:`onnxsim.gguf_reconstruct._gpt_oss_moe_ffn`.

Neither function is wired into :func:`onnxsim.reconstruct_gguf_graph`'s own
dispatch yet (see ``_gpt_oss_moe_ffn``'s own docstring) -- gpt-oss's
attention block (sliding-window pattern, attention sinks) has no home in
``_reconstruct_llama_family`` yet either -- so these tests call the two
functions directly rather than through ``onnxsim.reconstruct_gguf_graph``,
mirroring ``tests/test_gguf_reconstruct.py``'s own rigor (hand-encoded
byte-accurate synthetic GGUF v3 files, an independent from-scratch numpy
reference for everything checked, and a real onnxruntime run via
``onnx.utils.Extractor`` for the parts a bare ``sess.run`` can't reach
because of the MoE node -- see that file's own top comment) but adapted to
a function that has no dispatcher to build the surrounding graph for it.

Both test functions here build their own small ONNX graph directly with
``onnx.helper``/the module's own ``_Builder``, rather than via
``onnx.parser`` (this repo's usual preference for test-authored models --
see the repo's CLAUDE.md): the whole point of these tests is to exercise
the actual node-emission logic inside ``_interleave_gate_up``/
``_gpt_oss_moe_ffn``, which only exists as calls against ``_Builder``, so
there is no hand-written text graph to parse in the first place -- the
same situation ``test_gguf_reconstruct.py`` is in when it calls
``onnxsim.reconstruct_gguf_graph`` instead of authoring text.
"""

import struct

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.utils
import pytest

import onnxsim
from onnxsim.gguf_reconstruct import (
    _IR_VERSION,
    _OPSET,
    UnsupportedArchitectureError,
    _Builder,
    _gpt_oss_moe_ffn,
    _interleave_gate_up,
)
from onnxsim.onnx_simplifier import import_gguf_weights, read_gguf_metadata

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_METADATA_VALUE_TYPE_STRING = 8
GGML_TYPE_F32 = 0


def _string_bytes(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_STRING)
        + _string_bytes(value)
    )


def _align_up(n, align=32):
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def _write_gguf(path, kv_chunks, weights):
    """Minimal, byte-accurate GGUF v3 writer -- same technique as
    ``tests/test_gguf_reconstruct.py``'s own ``_write_gguf`` (every tensor
    plain F32, GGML's reversed-``ne`` convention over the array's own
    contiguous row-major bytes), duplicated here rather than imported since
    no test file in this repo imports fixtures from another one."""
    infos = b""
    data_chunks = []
    offset = 0
    for name, arr in weights.items():
        ne = list(reversed(arr.shape))
        raw = arr.astype("<f4").tobytes()
        infos += _string_bytes(name)
        infos += struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", GGML_TYPE_F32)
        infos += struct.pack("<Q", offset)
        data_chunks.append((offset, raw))
        offset = _align_up(offset + len(raw))

    header = struct.pack(
        "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(weights), len(kv_chunks)
    )
    body = b"".join(kv_chunks)
    header_end = len(header) + len(body) + len(infos)
    data_section_start = _align_up(header_end)

    with open(path, "wb") as f:
        f.write(header)
        f.write(body)
        f.write(infos)
        f.write(b"\x00" * (data_section_start - header_end))
        pos = data_section_start
        for rel_offset, raw in data_chunks:
            abs_offset = data_section_start + rel_offset
            f.write(b"\x00" * (abs_offset - pos))
            f.write(raw)
            pos = abs_offset + len(raw)


def _make_model(b, graph_inputs, graph_outputs, name):
    graph = onnx.helper.make_graph(
        b.nodes, name, graph_inputs, graph_outputs, initializer=b.initializers
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", _OPSET),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = _IR_VERSION
    return model


def test_gate_up_interleave_matches_independent_numpy_reference():
    """Standalone check of just :func:`_interleave_gate_up` -- both the 3D
    weight case (``[n_expert, n_ff, n_embd]`` pair -> ``[n_expert, 2*n_ff,
    n_embd]``) and the 2D bias case (``trailing=[]``) -- run for real via
    onnxruntime and compared to an independent numpy
    ``fused[:, 0::2, ...] = gate; fused[:, 1::2, ...] = up`` reference, not
    just re-derived by hand (see ``_interleave_gate_up``'s own docstring
    for the reshape/concat/reshape construction this exercises)."""
    ort = pytest.importorskip("onnxruntime")
    n_expert, n_ff, n_embd = 3, 5, 4

    b = _Builder()
    fc1_w = _interleave_gate_up(b, "gate", "up", n_expert, n_ff, [n_embd], "fc1_w")
    fc1_b = _interleave_gate_up(b, "gate_b", "up_b", n_expert, n_ff, [], "fc1_b")

    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            "gate", onnx.TensorProto.FLOAT, [n_expert, n_ff, n_embd]
        ),
        onnx.helper.make_tensor_value_info(
            "up", onnx.TensorProto.FLOAT, [n_expert, n_ff, n_embd]
        ),
        onnx.helper.make_tensor_value_info(
            "gate_b", onnx.TensorProto.FLOAT, [n_expert, n_ff]
        ),
        onnx.helper.make_tensor_value_info(
            "up_b", onnx.TensorProto.FLOAT, [n_expert, n_ff]
        ),
    ]
    graph_outputs = [
        onnx.helper.make_tensor_value_info(
            fc1_w, onnx.TensorProto.FLOAT, [n_expert, 2 * n_ff, n_embd]
        ),
        onnx.helper.make_tensor_value_info(
            fc1_b, onnx.TensorProto.FLOAT, [n_expert, 2 * n_ff]
        ),
    ]
    model = _make_model(b, graph_inputs, graph_outputs, "interleave_test")
    onnx.checker.check_model(model)

    rng = np.random.default_rng(0)
    gate = rng.standard_normal((n_expert, n_ff, n_embd)).astype(np.float32)
    up = rng.standard_normal((n_expert, n_ff, n_embd)).astype(np.float32)
    gate_b = rng.standard_normal((n_expert, n_ff)).astype(np.float32)
    up_b = rng.standard_normal((n_expert, n_ff)).astype(np.float32)

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    got_w, got_b = sess.run(
        [fc1_w, fc1_b], {"gate": gate, "up": up, "gate_b": gate_b, "up_b": up_b}
    )

    ref_w = np.empty((n_expert, 2 * n_ff, n_embd), dtype=np.float32)
    ref_w[:, 0::2, :] = gate
    ref_w[:, 1::2, :] = up
    ref_b = np.empty((n_expert, 2 * n_ff), dtype=np.float32)
    ref_b[:, 0::2] = gate_b
    ref_b[:, 1::2] = up_b

    np.testing.assert_array_equal(got_w, ref_w)
    np.testing.assert_array_equal(got_b, ref_b)


def _build_tiny_gpt_oss_moe_checkpoint(tmp_path, n_expert, n_expert_used, seed=5):
    """A hand-built, real GGUF v3 file holding just one gpt-oss-style MoE
    block's tensors (router + gate/up/down experts, all WITH the biases
    real gpt-oss checkpoints always carry -- see ``_gpt_oss_moe_ffn``'s
    docstring on why those are required, not optional, tensors for this
    architecture) -- deliberately not a full transformer checkpoint, since
    ``_gpt_oss_moe_ffn`` is not wired into any layer-building dispatch."""
    rng = np.random.default_rng(seed)
    n_embd, n_ff = 6, 4

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    weights = {
        "blk.0.ffn_gate_inp.weight": rand(n_expert, n_embd),
        "blk.0.ffn_gate_inp.bias": rand(n_expert),
        "blk.0.ffn_gate_exps.weight": rand(n_expert, n_ff, n_embd),
        "blk.0.ffn_gate_exps.bias": rand(n_expert, n_ff),
        "blk.0.ffn_up_exps.weight": rand(n_expert, n_ff, n_embd),
        "blk.0.ffn_up_exps.bias": rand(n_expert, n_ff),
        "blk.0.ffn_down_exps.weight": rand(n_expert, n_embd, n_ff),
        "blk.0.ffn_down_exps.bias": rand(n_expert, n_embd),
    }
    kv_chunks = [_kv_string("general.architecture", "gpt-oss")]
    path = str(tmp_path / "tiny_gpt_oss_moe.gguf")
    _write_gguf(path, kv_chunks, weights)
    config = dict(
        n_embd=n_embd, n_ff=n_ff, n_expert=n_expert, n_expert_used=n_expert_used
    )
    return path, weights, config


def _declare_closures(b, tensors):
    """A minimal stand-in for ``_reconstruct_llama_family``'s own
    ``declare``/``declare_optional`` closures -- deliberately simplified
    (every tensor here is plain F32, so no K-quant/MXFP4 dtype-mapping
    logic is needed, unlike the real closures) since these tests only need
    enough of that contract for ``_gpt_oss_moe_ffn`` to declare its own
    placeholder initializers correctly."""
    tensors_by_name = {t["name"]: t for t in tensors}

    def declare(name, expected_shape):
        info = tensors_by_name[name]
        assert info["shape"] == expected_shape, (name, info["shape"], expected_shape)
        b.placeholder_weight(name, expected_shape, onnx.TensorProto.FLOAT)
        return name

    def declare_optional(name, expected_shape):
        return declare(name, expected_shape) if name in tensors_by_name else None

    return declare, declare_optional


def test_gpt_oss_moe_ffn_wires_up_the_moe_node_correctly(tmp_path):
    """End-to-end: build a tiny gpt-oss-shaped GGUF checkpoint, run it
    through ``_gpt_oss_moe_ffn`` directly (bypassing
    ``reconstruct_gguf_graph``, which doesn't dispatch to it yet), and
    check (a) the ``com.microsoft.MoE`` node's own attributes/shapes/wiring
    match what ``_gpt_oss_moe_ffn``'s docstring claims, and (b) -- mirroring
    ``test_gguf_reconstruct.py``'s
    ``test_reconstructed_moe_graph_wires_up_the_moe_node_correctly`` --
    everything upstream of the MoE node itself (the router logits AND the
    interleaved fc1 weight/bias fusion) matches an independent numpy
    reference exactly, extracted via ``onnx.utils.Extractor`` and run
    through a real onnxruntime session, since a bare ``sess.run`` on the
    full model would build an execution plan that includes the (CPU-
    unsupported) swiglu MoE node regardless of which output is requested."""
    ort = pytest.importorskip("onnxruntime")
    n_expert, n_expert_used = 4, 2
    path, weights, config = _build_tiny_gpt_oss_moe_checkpoint(
        tmp_path, n_expert, n_expert_used
    )
    meta = read_gguf_metadata(path)

    n_embd, n_ff = config["n_embd"], config["n_ff"]
    batch, seq = 1, 3

    b = _Builder()
    declare, declare_optional = _declare_closures(b, meta["tensors"])
    h = "h"
    ffn_out = _gpt_oss_moe_ffn(
        b, h, "blk.0", n_embd, n_ff, n_expert, n_expert_used, declare, declare_optional
    )

    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            h, onnx.TensorProto.FLOAT, [batch, seq, n_embd]
        )
    ]
    graph_outputs = [
        onnx.helper.make_tensor_value_info(
            ffn_out, onnx.TensorProto.FLOAT, [batch, seq, n_embd]
        )
    ]
    model = _make_model(b, graph_inputs, graph_outputs, "gpt_oss_moe_test")
    onnx.checker.check_model(model)

    model, skipped = import_gguf_weights(model, path)
    assert skipped == []

    moe_nodes = [n for n in model.graph.node if n.op_type == "MoE"]
    assert len(moe_nodes) == 1
    node = moe_nodes[0]
    assert node.domain == "com.microsoft"
    attrs = {a.name: a for a in node.attribute}
    assert attrs["k"].i == n_expert_used
    assert attrs["activation_type"].s == b"swiglu"
    assert attrs["swiglu_fusion"].i == 1
    assert attrs["activation_alpha"].f == pytest.approx(1.702)
    assert attrs["activation_beta"].f == pytest.approx(1.0)
    assert attrs["swiglu_limit"].f == pytest.approx(7.0)
    assert attrs["normalize_routing_weights"].i == 1

    assert node.input[0] == h
    router_probs_name, fc1_w_name, fc1_b_name, down_w_name, down_b_name = node.input[1:]
    assert down_w_name == "blk.0.ffn_down_exps.weight"
    assert down_b_name == "blk.0.ffn_down_exps.bias"

    by_name = {i.name: i for i in model.graph.initializer}
    assert list(by_name["blk.0.ffn_gate_exps.weight"].dims) == [n_expert, n_ff, n_embd]
    assert list(by_name["blk.0.ffn_up_exps.weight"].dims) == [n_expert, n_ff, n_embd]
    assert list(by_name["blk.0.ffn_down_exps.weight"].dims) == [n_expert, n_embd, n_ff]
    assert list(by_name["blk.0.ffn_down_exps.bias"].dims) == [n_expert, n_embd]

    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    assert [d.dim_value for d in out_shape.dim] == [batch, seq, n_embd]

    # Extract the subgraph reachable from just the router logits and the
    # fused fc1 weight/bias -- all upstream of (never dependent on) the MoE
    # node, so onnx.utils.Extractor's real subgraph excludes it entirely
    # (unlike onnxruntime's own execution-plan builder, which would still
    # trip over it -- see this test's own docstring).
    extractable = onnx.ModelProto()
    extractable.CopyFrom(model)
    for name, shape in (
        (router_probs_name, [None, n_expert]),
        (fc1_w_name, [n_expert, 2 * n_ff, n_embd]),
        (fc1_b_name, [n_expert, 2 * n_ff]),
    ):
        extractable.graph.value_info.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        )
    probe_model = onnx.utils.Extractor(extractable).extract_model(
        [h], [router_probs_name, fc1_w_name, fc1_b_name]
    )

    rng = np.random.default_rng(9)
    h_val = rng.standard_normal((batch, seq, n_embd)).astype(np.float32)

    sess = ort.InferenceSession(
        probe_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    router_probs, fc1_w, fc1_b = sess.run(
        [router_probs_name, fc1_w_name, fc1_b_name], {h: h_val}
    )

    # Independent numpy reference, built from the very weights written into
    # the GGUF file above -- not by re-reading anything back out of `model`.
    router_w = weights["blk.0.ffn_gate_inp.weight"]
    router_b = weights["blk.0.ffn_gate_inp.bias"]
    ref_logits = h_val @ router_w.T + router_b
    ref_router_probs = ref_logits.reshape(-1, n_expert)

    gate_w, up_w = (
        weights["blk.0.ffn_gate_exps.weight"],
        weights["blk.0.ffn_up_exps.weight"],
    )
    gate_b, up_b = (
        weights["blk.0.ffn_gate_exps.bias"],
        weights["blk.0.ffn_up_exps.bias"],
    )
    ref_fc1_w = np.empty((n_expert, 2 * n_ff, n_embd), dtype=np.float32)
    ref_fc1_w[:, 0::2, :] = gate_w
    ref_fc1_w[:, 1::2, :] = up_w
    ref_fc1_b = np.empty((n_expert, 2 * n_ff), dtype=np.float32)
    ref_fc1_b[:, 0::2] = gate_b
    ref_fc1_b[:, 1::2] = up_b

    np.testing.assert_allclose(router_probs, ref_router_probs, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(fc1_w, ref_fc1_w)
    np.testing.assert_array_equal(fc1_b, ref_fc1_b)


def test_gpt_oss_moe_ffn_requires_both_or_neither_gate_up_bias(tmp_path):
    """A checkpoint with exactly one of ``ffn_gate_exps.bias``/
    ``ffn_up_exps.bias`` present can't build the fused fc1 bias
    ``swiglu_fusion=1`` needs (there is no defined meaning for interleaving
    a real bias with a missing one) -- must be rejected up front, the same
    way ``_reconstruct_llama_family``'s own ``declare`` rejects a
    shape-mismatched tensor rather than silently building a wrong graph."""
    n_expert, n_ff, n_embd = 2, 3, 4
    tensors = [
        {
            "name": "blk.0.ffn_gate_inp.weight",
            "shape": [n_expert, n_embd],
            "ggml_type": GGML_TYPE_F32,
        },
        {
            "name": "blk.0.ffn_gate_exps.weight",
            "shape": [n_expert, n_ff, n_embd],
            "ggml_type": GGML_TYPE_F32,
        },
        {
            "name": "blk.0.ffn_gate_exps.bias",
            "shape": [n_expert, n_ff],
            "ggml_type": GGML_TYPE_F32,
        },
        {
            "name": "blk.0.ffn_up_exps.weight",
            "shape": [n_expert, n_ff, n_embd],
            "ggml_type": GGML_TYPE_F32,
        },
        # blk.0.ffn_up_exps.bias deliberately omitted.
        {
            "name": "blk.0.ffn_down_exps.weight",
            "shape": [n_expert, n_embd, n_ff],
            "ggml_type": GGML_TYPE_F32,
        },
    ]
    b = _Builder()
    declare, declare_optional = _declare_closures(b, tensors)
    with pytest.raises(UnsupportedArchitectureError, match="ffn_gate_exps.bias"):
        _gpt_oss_moe_ffn(
            b, "h", "blk.0", n_embd, n_ff, n_expert, 1, declare, declare_optional
        )
