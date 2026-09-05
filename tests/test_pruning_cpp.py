"""Tests for ``onnxsim.prune_magnitude_cpp`` -- the C++-backed port of
``onnxsim.apply_magnitude_pruning`` (see ``onnxsim/passes/magnitude_pruning.h``).

Full parity with the pure-Python implementation: this port matches
MatMul/vanilla-Gemm, Conv (ordinary/depthwise/general-grouped), and
``com.microsoft`` Attention-family (``Attention``/
``DecoderMaskedSelfAttention``/``PackedAttention``) merged-QKV weights, over
FLOAT/FLOAT16/BFLOAT16, and offers both N:M semi-structured pruning and
``global_sparsity`` mode -- see that header's own doc comment.

``onnxsim.apply_magnitude_pruning`` (the pure-Python name) is now itself a
thin alias for :func:`onnxsim.prune_magnitude_cpp` (full parity verified --
see pruning.py's own "Magnitude pruning" section comment), so a test that
used to call BOTH entry points and compare their live outputs would be
tautological (literally the same code path twice) if left as-is. Those
instead compare the C++ port's output against a golden fixture captured
from the real pure-Python implementation *before* it was deleted -- see the
``_GOLDEN_*`` constants below and ``_golden_weight_bytes``'s own doc comment
for why these are the pruned weight tensor's own raw bytes (base64-encoded),
not a whole serialized ``ModelProto`` the way
``tests/test_transformer_block_pruning_cpp.py``'s own goldens are -- inlined
directly rather than as a checked-in ``.onnx``/fixture file: this repo's own
``.gitignore`` excludes ``*.onnx`` outright, and there is no existing
``tests/golden/``-style fixture-directory convention to follow instead --
see CLAUDE.md's own "Prefer onnx.parser-based model construction in tests"
note for the same "keep fixtures in the test file itself" spirit) --
preserving the original regression coverage (did the behavior change?)
without asserting a tautology. Every other test below asserts the expected
behavior directly (exact keep-counts, dtype round-trips, scope
in/exclusions) rather than comparing to a live Python call, so it stays a
real check after the alias too.
"""

import base64

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")
ml_dtypes = pytest.importorskip("ml_dtypes")


def _golden_weight_bytes(model, b64, node_index=0, input_index=1):
    """Compares a pruned model's weight tensor raw bytes against a frozen
    golden -- not a whole-``ModelProto`` byte comparison (the convention
    ``tests/test_transformer_block_pruning_cpp.py`` uses for its own golden
    fixtures): unlike a node-deletion rewrite, this pass's C++ port and the
    pure-Python original it now aliases always disagreed at the *graph*
    level even when numerically identical -- the pure-Python
    implementation mutated the existing initializer in place
    (``w_init.CopyFrom``), while the C++ port leaves the original
    initializer dangling and appends a new, anonymously-named one, rewiring
    the consuming node's input to it (see ``_weight``'s own doc comment) --
    so comparing full serialized bytes would fail on that structural
    difference alone, not a real regression. Comparing just the pruned
    weight's own raw bytes sidesteps that: dtype/shape are asserted
    separately by each test.
    """
    return _weight(model, node_index, input_index).tobytes() == base64.b64decode(b64)


def _model(body, initializer=(), opset=21, extra_opset=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}{extra_opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _single_conv_model(w, spatial=10, group=1):
    Cout, Cin_per_group, kh, kw = w.shape
    Cin = Cin_per_group * group
    out_spatial = spatial - kh + 1
    attrs = f"kernel_shape=[{kh},{kw}]"
    if group != 1:
        attrs += f", group={group}"
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          Y = Conv<{attrs}>(X, W1)
        }}
        """,
        initializer=[_f32(w, "W1")],
    )


def _weight(model, node_index=0, input_index=1):
    # Unlike the pure-Python apply_magnitude_pruning (which mutates the
    # existing initializer in place via w_init.CopyFrom), the C++ pass
    # leaves the original initializer dangling and appends a *new*,
    # anonymously-named one for the pruned weight, and rewires the node's
    # own weight input to it (matching every other onnxsim rewrite's
    # "replace, don't mutate" convention for constants) -- so the pruned
    # weight must be found via the node's current weight input name, not
    # assumed to still be named "W" or still be initializer[0].
    node = model.graph.node[node_index]
    w_name = node.input[input_index]
    init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(init)


def test_cpp_magnitude_pruning_reaches_target_sparsity():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    assert _weight(pruned).shape == _weight(model).shape


def test_cpp_magnitude_pruning_keeps_the_largest_entries_per_row():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.75)
    w = _weight(model).astype(np.float64)  # [K, N]
    w_pruned = _weight(pruned).astype(np.float64)
    for col in range(w.shape[1]):
        kept = np.flatnonzero(w_pruned[:, col] != 0)
        assert len(kept) == 16  # round(64 * 0.25)
        threshold = np.abs(w[:, col])[kept].min()
        dropped_max = np.abs(w[:, col])[np.flatnonzero(w_pruned[:, col] == 0)].max()
        assert dropped_max <= threshold


def test_cpp_magnitude_pruning_zero_sparsity_is_a_no_op():
    model = _matmul_model(K=32, N=8)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.0)
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_cpp_magnitude_pruning_rejects_invalid_sparsity():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(Exception):
        onnxsim.prune_magnitude_cpp(model, sparsity=1.0)


def test_cpp_magnitude_pruning_conv_reaches_target_sparsity():
    Cin, Cout = 4, 8  # K = Cin*3*3 = 36
    rng = np.random.default_rng(60)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)

    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    assert _weight(pruned).shape == w.shape


def test_cpp_magnitude_pruning_conv_depthwise_reaches_target_sparsity():
    C = 8
    rng = np.random.default_rng(61)
    w = rng.standard_normal((C, 1, 4, 4)).astype(np.float32)  # K=16, halved exactly
    model = _single_conv_model(w, spatial=10, group=C)

    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)


# Frozen from onnxsim.apply_magnitude_pruning's own real pure-Python
# implementation, on the exact tiny model each corresponding test below
# builds, before that implementation was deleted in favor of the C++ port
# (see this file's own module docstring).
_GOLDEN_SPARSITY_MATMUL = (
    "AAAAAAAAAAAAAAAA1P3jvpzKaL6M3P2+AAAAACyMKz9+Any+cdeevgAAAAAAAAAAAAAAACcz7r4A"
    "AAAAZP+xPjkPLL+dTGq+RFtzv5MPJb/5vWu/AAAAALA7Ir8AAAAAAAAAAAAAAACXEqG/AAAAAAAA"
    "AAAAAAAAfdtDvwAAAAA="
)


def test_cpp_magnitude_pruning_sparsity_matches_frozen_python_golden():
    rng = np.random.default_rng(7)
    K, N = 8, 4
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[_f32(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert _golden_weight_bytes(pruned, _GOLDEN_SPARSITY_MATMUL)


def test_cpp_magnitude_pruning_output_stays_finite_and_close():
    model = _matmul_model(K=64, N=16, seed=8)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.3)
    onnx.checker.check_model(pruned)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    sess_f = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sess_p = ort.InferenceSession(
        pruned.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_f,) = sess_f.run(None, {"X": x})
    (y_p,) = sess_p.run(None, {"X": x})
    assert np.all(np.isfinite(y_p))
    rel = np.linalg.norm(y_f - y_p) / max(np.linalg.norm(y_f), 1e-6)
    assert rel < 1.0  # 30% sparsity perturbs but shouldn't blow up the output


# --- N:M (semi-structured) pruning ----------------------------------------

_GOLDEN_NM_MATMUL = (
    "a54CQASQI8AAAAAAWVkRvwAAAAAAAAAAdEcBwAAAAACbfl2/BqxUQAAAAAAAAAAAAAAAAAAAAAAs"
    "D4e/ERfIvozB9j4AAAAArS91PwAAAAAAAAAAdd3FPwAAAACsVgG/AAAAAAAAAAD3sPc/AAAAAHJn"
    "eb7QS4A/AAAAAFtclb4="
)


def test_cpp_magnitude_pruning_nm_matches_frozen_python_golden():
    rng = np.random.default_rng(3)
    K, N = 8, 4
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[_f32(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, n=2, m=4)
    onnx.checker.check_model(pruned)
    assert _golden_weight_bytes(pruned, _GOLDEN_NM_MATMUL)


def test_cpp_magnitude_pruning_nm_keeps_n_of_every_m_columns():
    rng = np.random.default_rng(13)
    K, N = 64, 8  # cols=K=64, a multiple of m=4 -- no trailing partial group
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[_f32(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, n=2, m=4)
    w_nk = _weight(pruned).T.astype(np.float64)  # [N, K] -- output-channel-first
    for row in w_nk:
        for g in range(0, len(row), 4):
            assert np.count_nonzero(row[g : g + 4]) == 2


def test_cpp_magnitude_pruning_nm_tail_partial_group():
    # cols=K=10, m=4 -> two full groups of 4 (cols 0-3, 4-7) plus a
    # trailing partial group of 2 (cols 8-9), keep = min(2, max(1,
    # round(2*2/4))) = 1 -- mirrors pruning.py's own `_nm_mask` exactly.
    rng = np.random.default_rng(41)
    K, N = 10, 6
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[_f32(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, n=2, m=4)
    w_nk = _weight(pruned).T.astype(np.float64)  # [N, K]
    for row in w_nk:
        assert np.count_nonzero(row[0:4]) == 2
        assert np.count_nonzero(row[4:8]) == 2
        assert np.count_nonzero(row[8:10]) == 1


def test_cpp_magnitude_pruning_nm_conv():
    Cin, Cout = 4, 8  # cols = Cin*3*3 = 36
    rng = np.random.default_rng(31)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)
    pruned = onnxsim.prune_magnitude_cpp(model, n=2, m=4)
    onnx.checker.check_model(pruned)
    w_flat = _weight(pruned).reshape(Cout, -1).astype(np.float64)
    for row in w_flat:
        for g in range(0, 36, 4):
            assert np.count_nonzero(row[g : g + 4]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=2),
        dict(m=4),
        dict(n=0, m=4),
        dict(n=5, m=4),
        dict(n=2, m=4, global_sparsity=True),
    ],
)
def test_cpp_magnitude_pruning_nm_rejects_invalid_combinations(kwargs):
    model = _matmul_model(K=16, N=4)
    with pytest.raises(Exception):
        onnxsim.prune_magnitude_cpp(model, **kwargs)


# --- global_sparsity mode --------------------------------------------------

_GOLDEN_GLOBAL_SPARSITY_MIXED_MATMUL = (
    "AAAAAAAAAAAAAAAABFGgPwAAAACqAo2/53lMv4WD4z8AAAAAB4GXvwAAAAAAAAAA"
)

_GOLDEN_GLOBAL_SPARSITY_MIXED_CONV = "AAAAAMl37j8AAAAAuNfFv1MiwT8AAAAAAAAAAAAAAAA="


def test_cpp_magnitude_pruning_global_sparsity_matches_frozen_python_golden():
    rng = np.random.default_rng(51)
    K1, N1 = 6, 2
    Cin, Cout = 1, 2
    w1 = rng.standard_normal((K1, N1)).astype(np.float32)
    wc = rng.standard_normal((Cout, Cin, 2, 2)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K1}] X, float[b,{Cin},4,4] XC) => (float[batch,{N1}] Y, float[b,{Cout},3,3] YC)
        {{
          Y = MatMul(X, W1)
          YC = Conv<kernel_shape=[2,2]>(XC, WC)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(wc, "WC")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.6, global_sparsity=True)
    onnx.checker.check_model(pruned)
    assert _golden_weight_bytes(pruned, _GOLDEN_GLOBAL_SPARSITY_MIXED_MATMUL, node_index=0)
    assert _golden_weight_bytes(pruned, _GOLDEN_GLOBAL_SPARSITY_MIXED_CONV, node_index=1)


def test_cpp_magnitude_pruning_global_sparsity_pools_across_layers():
    # A layer whose weights are uniformly small should be pruned harder than
    # one whose weights are uniformly large, unlike the default per-layer
    # mode (which would zero the same *fraction* of each independently).
    rng = np.random.default_rng(101)
    K, N = 16, 4
    w_small = (rng.standard_normal((K, N)) * 1e-3).astype(np.float32)
    w_large = (rng.standard_normal((K, N)) * 1e3).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Ys, float[batch,{N}] Yl)
        {{
          Ys = MatMul(X, Ws)
          Yl = MatMul(X, Wl)
        }}
        """,
        initializer=[_f32(w_small, "Ws"), _f32(w_large, "Wl")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5, global_sparsity=True)
    w_small_pruned = _weight(pruned, node_index=0)
    w_large_pruned = _weight(pruned, node_index=1)
    total = w_small.size + w_large.size
    assert np.count_nonzero(w_small_pruned) + np.count_nonzero(
        w_large_pruned
    ) == round(total * 0.5)
    # The small-magnitude layer absorbs (much) more of the pruning than the
    # large-magnitude one -- no per-layer floor, unlike the default mode.
    assert np.count_nonzero(w_small_pruned) < np.count_nonzero(w_large_pruned)


def test_cpp_magnitude_pruning_global_sparsity_rejects_nm():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(Exception):
        onnxsim.prune_magnitude_cpp(model, n=2, m=4, global_sparsity=True)


def test_cpp_magnitude_pruning_global_sparsity_subgraph_aware():
    rng = np.random.default_rng(61)
    K, N = 16, 4
    w_then = rng.standard_normal((K, N)).astype(np.float32)
    w_else = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (bool cond, float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = If (cond) <
            then_branch = then_g () => (float[batch,{N}] Yt) {{ Yt = MatMul(X, Wt) }},
            else_branch = else_g () => (float[batch,{N}] Ye) {{ Ye = MatMul(X, We) }}
          >
        }}
        """
    )
    if_node = model.graph.node[0]
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            attr.g.initializer.append(_f32(w_then, "Wt"))
        elif attr.name == "else_branch":
            attr.g.initializer.append(_f32(w_else, "We"))
    onnx.checker.check_model(model)

    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5, global_sparsity=True)

    def _sub_weight(m, branch_name):
        node = m.graph.node[0]
        for attr in node.attribute:
            if attr.name == branch_name:
                sub = attr.g
                w_name = sub.node[0].input[1]
                init = next(t for t in sub.initializer if t.name == w_name)
                return onnx.numpy_helper.to_array(init)
        raise AssertionError(branch_name)

    total = w_then.size + w_else.size
    zeroed = np.count_nonzero(_sub_weight(pruned, "then_branch") == 0) + np.count_nonzero(
        _sub_weight(pruned, "else_branch") == 0
    )
    # Pooled across BOTH subgraphs combined -- not independently per branch.
    assert zeroed == round(total * 0.5)


# --- FLOAT16 / BFLOAT16 weight support -------------------------------------

_GOLDEN_FLOAT16_MATMUL = (
    "AABwOeY4FbQAAAAAjzQAAAAAZLtEOgAAAAAAAAAAAACZNgAAAAB8Nfa2D7oAAF21rruDtgAAxrj4"
    "uQAALjcAAA=="
)

_GOLDEN_BFLOAT16_MATMUL = (
    "vr4AAAAAAAAAAAU/AADQvgAAAAANPyS/qb7Xvl6/AAAAAL2+MT/SPqE+AAAAACq/nT4AAGK/AAAA"
    "AMg+AAAAAA=="
)


def test_cpp_magnitude_pruning_float16_matches_frozen_python_golden():
    rng = np.random.default_rng(11)
    K, N = 8, 4
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[onnx.numpy_helper.from_array(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    w_pruned = _weight(pruned)
    assert w_pruned.dtype == np.float16
    assert _golden_weight_bytes(pruned, _GOLDEN_FLOAT16_MATMUL)


def test_cpp_magnitude_pruning_bfloat16_matches_frozen_python_golden():
    rng = np.random.default_rng(11)
    K, N = 8, 4
    _ = rng.standard_normal((K, N))  # advance the stream to match the fp16 draw
    w = (rng.standard_normal((K, N)) * 0.5).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[onnx.numpy_helper.from_array(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    w_pruned = _weight(pruned)
    assert w_pruned.dtype == ml_dtypes.bfloat16
    assert _golden_weight_bytes(pruned, _GOLDEN_BFLOAT16_MATMUL)


def test_cpp_magnitude_pruning_float16_preserves_kept_bit_patterns():
    # Masking never recomputes a surviving value -- every kept entry's exact
    # original fp16 bit pattern should round-trip unchanged through the
    # float64 widen/narrow the C++ port uses internally.
    rng = np.random.default_rng(17)
    K, N = 32, 4
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{ Y = MatMul(X, W) }}
        """,
        initializer=[onnx.numpy_helper.from_array(w, "W")],
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    w_pruned = _weight(pruned)
    kept = w_pruned != 0
    np.testing.assert_array_equal(w_pruned[kept].view(np.uint16), w[kept].view(np.uint16))


# --- com.microsoft Attention-family merged-QKV-weight matching ------------

# The Attention/PackedAttention/DecoderMaskedSelfAttention goldens below are
# byte-identical to each other: same weight/bias, same seed, same
# sparsity=0.5 -- these three op types are matched by the exact same
# validation and masked by the exact same code path (see
# `MatchAttentionQkvWeightOnly`'s own doc comment), so an identical result
# is the expected outcome, not a copy-paste mistake.
_GOLDEN_ATTENTION_PLAIN = (
    "AAAAAEAK6D60MAm/LIgBPwAAAAAAAAAAFqt2vhRSpr4AAAAAqRWAPgAAAADbFEQ+SikCv75M8b4H"
    "qu4+AAAAALyrJz/l07k+tVidvvhqxT7wCkE+AAAAAHbde74AAAAAAAAAAOkXiT4AAAAAv2rzPgrs"
    "bb4AAAAAAAAAAAAAAABFcbW+ka+bPgAAAAAAAAAAAAAAADd3T74AAAAAAAAAAAAAAACedpc+AAAA"
    "AAAAAABf2p6+AAAAAN2f0b4AAAAAAAAAAAAAAAAAAAAAmdrNPtCx2j6OuLq+D1i0Pu1UBL/MoWG+"
    "AAAAAAAAAAAZ4qC+QvFdvgAAAAD7BFW+AAAAABetTD4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADH"
    "ZZ++XS9HPgAAAAAAAAAAAAAAAAAAAABaRdo+AAAAAAAAAAAAAAAAhN6Bvv1xwz5lRY4+dyxVPgAA"
    "AAAcM2M+u0+hPgAAAAAAAAAA4mkBvxN0kT4AAAAAU4WBvsHLiD4AAAAA"
)

_GOLDEN_PACKED_ATTENTION = _GOLDEN_ATTENTION_PLAIN

_GOLDEN_DECODER_MASKED_SELF_ATTENTION = _GOLDEN_ATTENTION_PLAIN


def _attention_qkv_weights():
    K = 8
    num_heads = 2
    Nq = Nk = Nv = 4
    total_n = Nq + Nk + Nv
    rng = np.random.default_rng(21)
    w = (rng.standard_normal((K, total_n)) * 0.3).astype(np.float32)
    b = (rng.standard_normal((total_n,)) * 0.1).astype(np.float32)
    return w, b, K, num_heads, Nq, Nk, Nv, total_n


def test_cpp_magnitude_pruning_attention_matches_frozen_python_golden():
    w, b, K, num_heads, Nq, Nk, Nv, total_n = _attention_qkv_weights()
    model = _model(
        f"""
        g (float[batch, seq, {K}] X, float[batch, seq, seq] mask) => (float[batch, seq, {Nq}] Y, float[2, batch, {num_heads}, seq, {Nv // num_heads}] present)
        {{
          Y, present = com.microsoft.Attention<num_heads={num_heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]>(X, W, B, mask)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
        extra_opset=', "com.microsoft": 1',
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert _golden_weight_bytes(pruned, _GOLDEN_ATTENTION_PLAIN)


def test_cpp_magnitude_pruning_packed_attention_matches_frozen_python_golden():
    w, b, K, num_heads, Nq, Nk, Nv, total_n = _attention_qkv_weights()
    model = _model(
        f"""
        g (float[batch, seq, {K}] X, float[batch, seq] tok_off, float[batch1] cum) => (float[batch, seq, {Nq}] Y)
        {{
          Y = com.microsoft.PackedAttention<num_heads={num_heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]>(X, W, B, tok_off, cum)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
        extra_opset=', "com.microsoft": 1',
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert _golden_weight_bytes(pruned, _GOLDEN_PACKED_ATTENTION)


def test_cpp_magnitude_pruning_decoder_masked_self_attention_matches_frozen_python_golden():
    w, b, K, num_heads, Nq, Nk, Nv, total_n = _attention_qkv_weights()
    past_shape = f"2, batch, {num_heads}, past_seq, {Nv // num_heads}"
    model = _model(
        f"""
        g (float[batch, 1, {K}] X, float[{past_shape}] past) => (float[batch, 1, {Nq}] Y, float[{past_shape}] present)
        {{
          Y, present = com.microsoft.DecoderMaskedSelfAttention<num_heads={num_heads}>(X, W2, B2, , past)
        }}
        """,
        initializer=[_f32(w, "W2"), _f32(b, "B2")],
        extra_opset=', "com.microsoft": 1',
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert _golden_weight_bytes(pruned, _GOLDEN_DECODER_MASKED_SELF_ATTENTION)


def test_cpp_magnitude_pruning_decoder_masked_self_attention_declines_constant_past():
    # A constant `past` has no established/tested slicing path -- declined
    # conservatively (mirrors pruning.py's own `_match_attention_producer`),
    # even though this pass never touches `past` at all either way.
    w, b, K, num_heads, Nq, Nk, Nv, total_n = _attention_qkv_weights()
    rng = np.random.default_rng(81)
    past_const = rng.standard_normal((2, 1, num_heads, 1, Nv // num_heads)).astype(
        np.float32
    )
    model = _model(
        f"""
        g (float[1, 1, {K}] X) => (float[1, 1, {Nq}] Y, float[2, 1, {num_heads}, 1, {Nv // num_heads}] present)
        {{
          Y, present = com.microsoft.DecoderMaskedSelfAttention<num_heads={num_heads}>(X, W2, B2, , PastConst)
        }}
        """,
        initializer=[_f32(w, "W2"), _f32(b, "B2"), _f32(past_const, "PastConst")],
        extra_opset=', "com.microsoft": 1',
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), w)


def test_cpp_magnitude_pruning_attention_declines_uneven_qkv_split():
    w, b, K, num_heads, Nq, Nk, Nv, total_n = _attention_qkv_weights()
    # num_heads doesn't evenly divide Nq -- malformed, declined rather than
    # guessed at.
    model = _model(
        f"""
        g (float[batch, seq, {K}] X) => (float[batch, seq, {Nq}] Y, float[2, batch, {num_heads}, seq, {Nv // num_heads}] present)
        {{
          Y, present = com.microsoft.Attention<num_heads=3, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]>(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
        extra_opset=', "com.microsoft": 1',
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), w)


def test_cpp_magnitude_pruning_ignores_unrelated_domain_attention_op():
    # A same-named "Attention" node in a DIFFERENT domain is not
    # com.microsoft's -- must never match.
    w, b, K, num_heads, Nq, Nk, Nv, total_n = _attention_qkv_weights()
    model = _model(
        f"""
        g (float[batch, seq, {K}] X) => (float[batch, seq, {Nq}] Y)
        {{
          Y = custom.domain.Attention<num_heads={num_heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]>(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
        extra_opset=', "custom.domain": 1',
    )
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), w)
