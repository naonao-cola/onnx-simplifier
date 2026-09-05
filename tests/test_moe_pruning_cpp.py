"""Tests for ``onnxsim.apply_moe_expert_channel_pruning_cpp`` -- the C++-backed
port of ``onnxsim.apply_moe_expert_channel_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s "MoE (com.microsoft::MoE)
expert-intermediate-channel pruning" section). Data-free/structural only --
whole-expert pruning (``onnxsim.apply_moe_whole_expert_pruning``, which needs
runtime calibration data via an ``onnxruntime.InferenceSession`` observing
router activations) is NOT ported, matching this codebase's established
C++-port scope decision (this build has no ONNX Runtime linked into it at
all -- see the repo's own CLAUDE.md).

Tests here are adapted from ``tests/test_pruning.py``'s own
``apply_moe_expert_channel_pruning`` coverage (search
"MoE expert-intermediate-channel pruning" there) -- NOT its separate
"MoE whole-expert pruning" section, which is out of scope here. Every test
that actually prunes something runs the result through a real onnxruntime
CPU session, exactly like the Python-side tests do: confirmed, empirically,
to be the one real oracle available for ``com.microsoft::MoE`` in this
environment.

``onnxsim.apply_moe_expert_channel_pruning`` (the pure-Python name) is now
itself a thin alias for :func:`onnxsim.apply_moe_expert_channel_pruning_cpp`
(full parity verified, including FLOAT16/BFLOAT16 weights -- see
pruning.py's own "MoE expert-intermediate-channel pruning" section
comment), so the one test below that used to call BOTH entry points and
compare their live outputs would be tautological (literally the same code
path twice) if left as-is -- it now instead compares the C++ port's output
against a golden fixture captured from the real pure-Python implementation
*before* it was deleted (see ``_GOLDEN_*`` below, base64-encoded serialized
``ModelProto`` bytes, inlined directly per this repo's own established
convention -- see ``tests/test_transformer_block_pruning_cpp.py``'s own
identical precedent/module-docstring for the full rationale).
"""

import base64

import ml_dtypes
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _golden(b64):
    return onnx.load_from_string(base64.b64decode(b64))


# Frozen from onnxsim.apply_moe_expert_channel_pruning's own real pure-Python
# implementation, on the exact model + seed
# test_moe_expert_channel_pruning_cpp_matches_python_port builds, before
# that implementation was deleted in favor of the C++ port (see this file's
# own module docstring).
_GOLDEN_MOE_EXPERT_CHANNEL_MATCHES_PYTHON_PORT = (
    "CAo6qQkKbQoBWAoBUgoERkMxVwoERkMxQgoERkMyVwoAEgFZIgNNb0UqCAoBaxgCoAECKhoKD2Fj"
    "dGl2YXRpb25fdHlwZSIEcmVsdaABAyoUCg1zd2lnbHVfZnVzaW9uGACgAQI6DWNvbS5taWNyb3Nv"
    "ZnQSAWcq0QMIBAgECAcQAUIERkMxV0rAA5QdML/aq6g/2tsEv+b3Xb/aCok/c3p1v4mni79qccg/"
    "wlsMP0PY3r7H8Qm/Sjixv3WokT/S8JA+gcImv0W7uDwU7uM/os7FvwkMHMCNPCK+zVVcP36w0b7e"
    "yxrAdMpmv6urZr8R6/K/yoooPzTAkj6qi9u9m0lOv4zuaz6z0pe98L1DPjJDcr7hr+w+oPKavSsE"
    "9b/39iE/bp6pPi79O78btnY/cnfpPuUwEb4o1PA+QuiHP84iTT86rqI/LgRqvyUg1b9e5BG/xAS+"
    "v0I8uj4oOxw/AMD3PxPAcr+qwRm/MhpmP201Qz9yFQ0/TJ7QvjXLmr+6XTg/2kU9P7x2L727O+e+"
    "COAiPrYzKj73m8G/vSHvvvp8FT8KOOK+3U48Pk9i9r5QOlq/7/1HwHM/Bb9QBES/kYBCv8ELc7+P"
    "foG/WlPGPxX0Cz/8f5w+RfxyPxSeiD4FgNs/TeGFPlKTJ79FzTU/R641v2uihz/6E7c/LpbZPW6G"
    "wL976SHAh8KUP1Ae8j7hw+s+ojzrvPwNlb45cMs/MeauP2sjiD4eivQ/aa3dPhfiDT+DoB8/ObzX"
    "vhcXJz++Rhc/PeFevxIC3L8q0QMIBAgHCAQQAUIERkMyV0rAAwdW0T30Soo/dhLcvxozSD/PFbk/"
    "ShG4v0RcnL+bu88/iMyfvmH+ZT9Ly+k/RRiGP25pHT/OfAO+XtGnP0uxPD6tVAo+6ZcDP3s6sz59"
    "Agq/4R9RvxsOwb58872/zvmOvhCveb+IxII9sfJ0vmZPqr6t1P0/vxmDPjQ4DUCFvWC+du2ev8C2"
    "dL/154S/Bq23PyqDIz98n1a/cee5vz3AHz/HZvY+SQpJv3MQzb0vWsQ/p7jnv0vFiD9w8mY+ZIoX"
    "P43vmj9ASBc+WkzkvsMlJT/BgmI/cX8SP6mYnD3NuDK/qOAIQLn/cr9cRKk+kqZgv8h4K78gDO4/"
    "8g3Pvl2Ugz/v2aO/sOylPVmpPz6BJbq+oxdFPlg3rz2LgGS/g5oIP3XDTj/k7K6+aawkv3B/8T4o"
    "WcQ+b9D4vinHDECruBm+b2lLvxshe7/hcWc//cg/vasnBECo6Pk+VmPCPkvjZz5Zme0+Ozihv76r"
    "bT8p872/9B4PPqJJ/z6mV7e+Iw1VP91aob+P4fK8WR16P3zYgD46nam/XoJtP8U8TL/pWta/j866"
    "vzigFMCCHck9eXLcP81cH79iSos/gVlQP7hQgb4qTggECAQQAUIERkMxQkpA1aqNP7Kh7r69uOi+"
    "DS6yvka8ET/VSt6+5QO/vmRaNb5dIoO/0WefP9THsr/LLpo/H02AP2ZcGr/7/ii/D7thv1oTCgFY"
    "Eg4KDAgBEggKAggGCgIIB1oTCgFSEg4KDAgBEggKAggGCgIIBGITCgFZEg4KDAgBEggKAggGCgII"
    "B0IECgAQEkIRCg1jb20ubWljcm9zb2Z0EAE="
)


def _model(body, initializer=(), opset=21):
    # Pinning ir_version: 10, same as tests/test_pruning.py's own _model --
    # matches the older onnxruntime bundled with some CI wheels (which cap
    # at IR version 11); _run below runs these models through onnxruntime.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _moe_model(
    fc1_w,
    fc2_w,
    fc1_b=None,
    fc2_b=None,
    fc3_w=None,
    activation="relu",
    swiglu_fusion=0,
    k=2,
    tokens=6,
):
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    fc2_b_arg = "FC2B" if fc2_b is not None else ""
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    model = _model(
        f"""
        g (float[{tokens},{hidden}] X, float[{tokens},{num_experts}] R) => (float[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}", swiglu_fusion={swiglu_fusion}> (X, R, FC1W, {fc1_b_arg}, FC2W, {fc2_b_arg}, {fc3_w_arg})
        }}
        """,
        opset=18,
    )
    inits = [_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")]
    if fc1_b is not None:
        inits.append(_f32(fc1_b, "FC1B"))
    if fc2_b is not None:
        inits.append(_f32(fc2_b, "FC2B"))
    if fc3_w is not None:
        inits.append(_f32(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _moe_inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def test_moe_expert_channel_pruning_cpp_matches_ort_masking_oracle():
    # Physically removing the lowest-importance `inter_size` channels must be
    # numerically identical to *zeroing* those same channels (fc1's own row,
    # fc1_experts_bias's own entry, fc2's own column) in a same-shape model --
    # the real onnxruntime CPU-execution oracle this pass's own safety
    # argument rests on (mirrors
    # test_pruning.py::test_moe_expert_channel_pruning_matches_ort_masking_oracle).
    E, hidden, inter = 5, 10, 12
    rng = np.random.default_rng(3)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.5).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.5).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    fc2_b = rng.standard_normal((E, hidden)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, fc1_b=fc1_b, fc2_b=fc2_b)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (E, 6, hidden)
    assert inits["FC2W"].shape == (E, hidden, 6)
    assert inits["FC1B"].shape == (E, 6)
    np.testing.assert_array_equal(
        inits["FC2B"], fc2_b
    )  # indexes hidden_size, untouched

    sq = (
        np.sum(fc1_w**2, axis=(0, 2))
        + np.sum(fc2_w**2, axis=(0, 1))
        + np.sum(fc1_b**2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])
    drop = np.setdiff1d(np.arange(inter), keep)
    np.testing.assert_allclose(inits["FC1W"], fc1_w[:, keep, :])
    np.testing.assert_allclose(inits["FC2W"], fc2_w[:, :, keep])
    np.testing.assert_allclose(inits["FC1B"], fc1_b[:, keep])

    fc1_w_masked = fc1_w.copy()
    fc1_w_masked[:, drop, :] = 0
    fc1_b_masked = fc1_b.copy()
    fc1_b_masked[:, drop] = 0
    fc2_w_masked = fc2_w.copy()
    fc2_w_masked[:, :, drop] = 0
    masked = _moe_model(fc1_w_masked, fc2_w_masked, fc1_b=fc1_b_masked, fc2_b=fc2_b)

    rng2 = np.random.default_rng(7)
    tokens = 6
    feeds = {
        "X": rng2.standard_normal((tokens, hidden)).astype(np.float32),
        "R": rng2.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-5, atol=1e-5)


def test_moe_expert_channel_pruning_cpp_adversarial_conflicting_fc1_fc2_importance():
    # Deliberately conflicting per-channel importance: channel A has a large
    # fc1 row but a tiny fc2 column, channel B the reverse, and channel C is
    # tiny on both. A bug that ranked by only one of fc1/fc2 (instead of the
    # documented combined root-sum-square of both) would keep A or B, not
    # both -- this catches that by making the *combined* score of A and B
    # comparably large while C is small on both, so only C should be dropped
    # at sparsity=1/3.
    E, hidden, inter = 3, 4, 3
    rng = np.random.default_rng(11)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.01).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.01).astype(np.float32)
    fc1_w[:, 0, :] = 5.0  # channel 0 (A): large fc1, tiny fc2 (left as noise)
    fc2_w[:, :, 1] = 5.0  # channel 1 (B): tiny fc1 (noise), large fc2
    # channel 2 (C) stays small noise on both -- the one channel expected to
    # be dropped.
    model = _moe_model(fc1_w, fc2_w)

    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=1.0 / 3.0)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (E, 2, hidden)
    np.testing.assert_allclose(inits["FC1W"], fc1_w[:, [0, 1], :])
    np.testing.assert_allclose(inits["FC2W"], fc2_w[:, :, [0, 1]])


def test_moe_expert_channel_pruning_cpp_zero_sparsity_is_a_no_op():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(13)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w)
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.0)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_invalid_sparsity_raises():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(97)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w)
    with pytest.raises(Exception):
        onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=1.0)
    with pytest.raises(Exception):
        onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=-0.1)


def test_moe_expert_channel_pruning_cpp_declines_fc3():
    # com.microsoft::MoE's own CPU execution provider, in this environment,
    # raises "FC3 is not implemented for CPU MoE" for any activation_type --
    # confirmed empirically, see pruning.py's own section comment -- so a
    # node with fc3_experts_weights present is left completely untouched
    # rather than pruned against a shape this environment has no real
    # runtime to validate.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(17)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, fc3_w=fc3_w, activation="silu")
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)
    np.testing.assert_array_equal(inits["FC3W"], fc3_w)


def test_moe_expert_channel_pruning_cpp_declines_swiglu_activation():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(19)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, activation="swiglu")
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_declines_nonzero_swiglu_fusion():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(23)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, swiglu_fusion=1)
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_declines_fused_swiglu_shape():
    # A real fused-swiglu fc1 doubles its own row count (fusion_size=2) --
    # fc1's axis-1 size then never equals fc2's own axis-2 size, so this
    # declines via the shape-consistency check alone, without even needing
    # to read swiglu_fusion/activation_type.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(29)
    fc1_w = rng.standard_normal((E, 2 * inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, activation="swiglu", swiglu_fusion=1)
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_declines_tied_fc1_weight():
    # fc1_experts_weights reused by a second node -- an in-place resize
    # would corrupt that other consumer, so this is declined outright, the
    # same tied-weight guard every other chain-matcher in this codebase
    # applies via its own consumer map.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(31)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _model(
        f"""
        g (float[6,{hidden}] X, float[6,{E}] R) => (float[6,{hidden}] Y, float[{E},{inter},{hidden}] Z)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, , FC2W)
          Z = Identity(FC1W)
        }}
        """,
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_declines_non_constant_weight():
    # fc1_experts_weights fed by a graph input (not an initializer) --
    # there's nothing to slice in place, so the node is left untouched
    # rather than raising.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(37)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _model(
        f"""
        g (float[6,{hidden}] X, float[6,{E}] R, float[{E},{inter},{hidden}] FC1W) => (float[6,{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_declines_mismatched_hidden_size():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(41)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden + 1, inter)).astype(np.float32)  # mismatched
    model = _model(
        f"""
        g (float[6,{hidden}] X, float[6,{E}] R) => (float[6,{hidden + 1}] Y)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_cpp_no_bias_matches_ort_masking_oracle():
    E, hidden, inter = 4, 8, 10
    rng = np.random.default_rng(43)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.5).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.5).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, activation="gelu")
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.4)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    keep_count = inits["FC1W"].shape[1]
    assert keep_count == 6

    sq = np.sum(fc1_w**2, axis=(0, 2)) + np.sum(fc2_w**2, axis=(0, 1))
    keep = np.sort(np.argsort(-np.sqrt(sq))[:keep_count])
    drop = np.setdiff1d(np.arange(inter), keep)
    fc1_w_masked = fc1_w.copy()
    fc1_w_masked[:, drop, :] = 0
    fc2_w_masked = fc2_w.copy()
    fc2_w_masked[:, :, drop] = 0
    masked = _moe_model(fc1_w_masked, fc2_w_masked, activation="gelu")

    rng2 = np.random.default_rng(47)
    tokens = 6
    feeds = {
        "X": rng2.standard_normal((tokens, hidden)).astype(np.float32),
        "R": rng2.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_moe_expert_channel_pruning_cpp_multiple_nodes_pruned_independently():
    E1, hidden1, inter1 = 3, 4, 6
    E2, hidden2, inter2 = 2, 5, 8
    rng = np.random.default_rng(53)
    fc1_w1 = rng.standard_normal((E1, inter1, hidden1)).astype(np.float32)
    fc2_w1 = rng.standard_normal((E1, hidden1, inter1)).astype(np.float32)
    fc1_w2 = rng.standard_normal((E2, inter2, hidden2)).astype(np.float32)
    fc2_w2 = rng.standard_normal((E2, hidden2, inter2)).astype(np.float32)
    model = _model(
        f"""
        g (float[6,{hidden1}] X1, float[6,{E1}] R1, float[6,{hidden2}] X2, float[6,{E2}] R2)
            => (float[6,{hidden1}] Y1, float[6,{hidden2}] Y2)
        {{
          Y1 = com.microsoft.MoE <k=1, activation_type="relu"> (X1, R1, FC1W1, , FC2W1)
          Y2 = com.microsoft.MoE <k=1, activation_type="relu"> (X2, R2, FC1W2, , FC2W2)
        }}
        """,
        initializer=[
            _f32(fc1_w1, "FC1W1"),
            _f32(fc2_w1, "FC2W1"),
            _f32(fc1_w2, "FC1W2"),
            _f32(fc2_w2, "FC2W2"),
        ],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W1"].shape == (E1, 3, hidden1)
    assert inits["FC1W2"].shape == (E2, 4, hidden2)


def test_moe_expert_channel_pruning_cpp_matches_python_port():
    # The C++ port and the pure-Python reference implementation must agree
    # bit-for-bit on a plain (in-scope) MoE node: same importance ranking,
    # same keep set, same sliced values. `inter` is deliberately even so
    # `inter * sparsity` is an exact integer, not a .5 tie -- Python's
    # `round()` (round-half-to-even) and C++'s `std::llround()` (round-half-
    # away-from-zero) can legitimately disagree exactly at such a tie (the
    # same tie-breaking caveat TopKIndicesAscending's own comment and
    # test_attention_head_pruning_cpp.py's own cross-check test -- which
    # compares execution OUTPUT, not raw tensors, for exactly this reason --
    # already document), which is not a bug in either port.
    #
    # ``apply_moe_expert_channel_pruning`` is now itself an alias for the
    # C++ port (see this file's own module docstring), so this compares
    # against a golden fixture frozen from the real pure-Python
    # implementation instead of calling it live (which would make this
    # tautological).
    E, hidden, inter = 4, 7, 8
    rng = np.random.default_rng(59)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, fc1_b=fc1_b)

    golden = _golden(_GOLDEN_MOE_EXPERT_CHANNEL_MATCHES_PYTHON_PORT)
    pruned_cpp = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits_golden = _moe_inits(golden)
    inits_cpp = _moe_inits(pruned_cpp)
    np.testing.assert_array_equal(inits_golden["FC1W"], inits_cpp["FC1W"])
    np.testing.assert_array_equal(inits_golden["FC2W"], inits_cpp["FC2W"])
    np.testing.assert_array_equal(inits_golden["FC1B"], inits_cpp["FC1B"])


def test_moe_expert_channel_pruning_cpp_prunes_inside_if_subgraph():
    # Subgraph-aware (IterSubgraphs, see structured_pruning_entry.cpp's own
    # "Subgraph recursion" section comment): a com.microsoft::MoE node nested
    # inside an `If` node's `then_branch` is matched and pruned exactly as if
    # that subgraph were its own top-level graph.
    # `inter` deliberately even -- see test_moe_expert_channel_pruning_cpp_
    # matches_python_port's own comment on why sparsity=0.5 needs an even
    # `inter` for an unambiguous (non-tied) expected keep_count.
    E, hidden, inter = 3, 6, 6
    rng = np.random.default_rng(61)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)

    then_graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "MoE",
                ["X", "R", "FC1W", "", "FC2W"],
                ["Y"],
                domain="com.microsoft",
                k=1,
                activation_type="relu",
            )
        ],
        "then_graph",
        [],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [6, hidden])],
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")],
    )
    else_graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["X"], ["Y"])],
        "else_graph",
        [],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [6, hidden])],
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "If",
                    ["Cond"],
                    ["Out"],
                    then_branch=then_graph,
                    else_branch=else_graph,
                )
            ],
            "g",
            [
                onnx.helper.make_tensor_value_info("Cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info(
                    "X", onnx.TensorProto.FLOAT, [6, hidden]
                ),
                onnx.helper.make_tensor_value_info("R", onnx.TensorProto.FLOAT, [6, E]),
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "Out", onnx.TensorProto.FLOAT, [6, hidden]
                )
            ],
        ),
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    # `If`'s own then/else branches implicitly capture `X`/`R` from the
    # enclosing graph's own inputs -- valid ONNX (the same "implicit
    # capture" shape structured_pruning_entry.cpp's own "Subgraph recursion"
    # section comment documents), so this model is deliberately left without
    # a checker call (the checker requires every implicitly-captured name to
    # actually resolve in an enclosing scope at check time, which is
    # satisfied here, but is beside this test's own point).

    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    then_attr = next(
        a for a in pruned.graph.node[0].attribute if a.name == "then_branch"
    )
    fc1w_pruned = onnx.numpy_helper.to_array(
        next(t for t in then_attr.g.initializer if t.name == "FC1W")
    )
    assert fc1w_pruned.shape == (E, 3, hidden)


# --- FP16/BFloat16 weight support -------------------------------------------
#
# MatchMoeProducer (structured_pruning_entry.cpp) used to hard-require
# ``onnx.TensorProto.FLOAT`` for ``fc1_experts_weights``/
# ``fc2_experts_weights``/``fc1_experts_bias``, silently declining any MoE
# node whose weights were stored as FLOAT16 or BFLOAT16 -- narrower than
# ``onnxsim.apply_moe_expert_channel_pruning``'s own ``_is_supported_float_
# dtype`` (see that module's own "FP16/BFloat16 weight support" section
# comment). Now widened via IsSupportedFloatDtype/ReadTensorAsF64/
# WriteF64TensorAs (see structured_pruning_entry.cpp's own "MoE
# expert-intermediate-channel pruning" section top comment) -- these two
# tests mirror tests/test_pruning.py's own
# ``test_moe_expert_channel_pruning_fp16_matches_ort_masking_oracle``/
# BFLOAT16 array-oracle tests, against the C++-backed entry point.
#
# FLOAT16: onnxruntime's CPU MoE kernel genuinely executes FLOAT16 (confirmed
# separately, same as the pure-Python test suite), so this runs a real
# session. BFLOAT16 has no onnxruntime CPU execution support in this
# environment at all (a plain BFLOAT16 MatMul session raises NOT_IMPLEMENTED
# at session-creation time) -- so that test checks correctness at the array
# level (dtype preservation, exact per-element bfloat16 decode) instead.


def test_moe_expert_channel_pruning_cpp_fp16_matches_ort_masking_oracle():
    E, hidden, inter, tokens, k = 5, 10, 12, 6, 2
    rng = np.random.default_rng(1009)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float16)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float16)

    model = _model(
        f"""
        g (float16[{tokens},{hidden}] X, float16[{tokens},{E}] R) => (float16[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k={k}, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f16(fc1_w, "FC1W"), _f16(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits["FC1W"].data_type == onnx.TensorProto.FLOAT16
    assert list(inits["FC1W"].dims) == [E, 6, hidden]

    fc1_w64 = fc1_w.astype(np.float64)
    fc2_w64 = fc2_w.astype(np.float64)
    sq = np.sum(fc1_w64**2, axis=(0, 2)) + np.sum(fc2_w64**2, axis=(0, 1))
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])

    fc1_w_pruned = onnx.numpy_helper.to_array(inits["FC1W"])
    fc2_w_pruned = onnx.numpy_helper.to_array(inits["FC2W"])
    # Exact bit-pattern match (not assert_allclose) -- this pass only ever
    # slices/reorders existing rows/columns, never recomputes a surviving
    # value, so the round trip through float64 must reproduce the original
    # fp16 bits exactly.
    np.testing.assert_array_equal(
        fc1_w_pruned.view(np.uint16), fc1_w[:, keep, :].view(np.uint16)
    )
    np.testing.assert_array_equal(
        fc2_w_pruned.view(np.uint16), fc2_w[:, :, keep].view(np.uint16)
    )

    rng2 = np.random.default_rng(1010)
    x = rng2.standard_normal((tokens, hidden)).astype(np.float16)
    r = rng2.standard_normal((tokens, E)).astype(np.float16)
    (y_pruned,) = _run(pruned, {"X": x, "R": r})
    assert y_pruned.dtype == np.float16

    drop = np.setdiff1d(np.arange(inter), keep)
    fc1_masked = fc1_w64.copy()
    fc1_masked[:, drop, :] = 0.0
    fc2_masked = fc2_w64.copy()
    fc2_masked[:, :, drop] = 0.0
    masked_model = _model(
        f"""
        g (float16[{tokens},{hidden}] X, float16[{tokens},{E}] R) => (float16[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k={k}, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[
            _f16(fc1_masked.astype(np.float32), "FC1W"),
            _f16(fc2_masked.astype(np.float32), "FC2W"),
        ],
        opset=18,
    )
    masked_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    (y_masked,) = _run(masked_model, {"X": x, "R": r})
    np.testing.assert_allclose(
        y_pruned.astype(np.float64), y_masked.astype(np.float64), rtol=5e-2, atol=5e-2
    )


def test_moe_expert_channel_pruning_cpp_bfloat16_preserves_dtype_and_matches_array_oracle():
    E, hidden, inter = 5, 10, 12
    rng = np.random.default_rng(1011)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(ml_dtypes.bfloat16)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(ml_dtypes.bfloat16)
    fc1_b = rng.standard_normal((E, inter)).astype(ml_dtypes.bfloat16)

    model = _model(
        f"""
        g (bfloat16[batch,{hidden}] X, bfloat16[batch,{E}] R) => (bfloat16[batch,{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, FC1B, FC2W)
        }}
        """,
        initializer=[_bf16(fc1_w, "FC1W"), _bf16(fc1_b, "FC1B"), _bf16(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_moe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits["FC1W"].data_type == onnx.TensorProto.BFLOAT16
    assert inits["FC2W"].data_type == onnx.TensorProto.BFLOAT16
    assert inits["FC1B"].data_type == onnx.TensorProto.BFLOAT16

    fc1_w64 = fc1_w.astype(np.float64)
    fc2_w64 = fc2_w.astype(np.float64)
    fc1_b64 = fc1_b.astype(np.float64)
    sq = (
        np.sum(fc1_w64**2, axis=(0, 2))
        + np.sum(fc2_w64**2, axis=(0, 1))
        + np.sum(fc1_b64**2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])

    fc1_w_pruned = onnx.numpy_helper.to_array(inits["FC1W"])
    fc2_w_pruned = onnx.numpy_helper.to_array(inits["FC2W"])
    fc1_b_pruned = onnx.numpy_helper.to_array(inits["FC1B"])
    assert fc1_w_pruned.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(
        fc1_w_pruned.view(np.uint16), fc1_w[:, keep, :].view(np.uint16)
    )
    np.testing.assert_array_equal(
        fc2_w_pruned.view(np.uint16), fc2_w[:, :, keep].view(np.uint16)
    )
    np.testing.assert_array_equal(
        fc1_b_pruned.view(np.uint16), fc1_b[:, keep].view(np.uint16)
    )
