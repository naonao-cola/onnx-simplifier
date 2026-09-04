"""Tests for ``onnxsim.apply_transformer_block_pruning_cpp`` -- the C++-backed
port of ``onnxsim.apply_transformer_block_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s "Transformer block (depth) pruning"
section and ``ApplyTransformerBlockPruning``). Like
``onnxsim.apply_structured_wanda_pruning_cpp``, this runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor. Unlike every other C++-backed pruning entry point, this one
performs real graph surgery (node deletion + consumer rewiring), not tensor
slicing, so several of these tests confirm the rewiring itself is correct
via an independently hand-built "already pruned" reference model, not merely
that *some* pruning happened.

See ``tests/test_pruning.py``'s own "apply_transformer_block_pruning" section
for the exact matched pattern (a pre-norm residual sub-block whose merge is a
bare ``Add`` and whose entry norm is a plain LayerNormalization/
RMSNormalization/SimplifiedLayerNormalization node or a fused
SkipLayerNormalization-family node's own optional fourth output) and the full
set of decline conditions this file's tests cross-check against.

``onnxsim.apply_transformer_block_pruning`` (the pure-Python name) is now
itself a thin alias for :func:`onnxsim.apply_transformer_block_pruning_cpp`
(full parity verified -- see pruning.py's own "Transformer block (depth)
pruning" section comment), so the handful of tests below that used to call
BOTH entry points and compare their live outputs would be tautological
(literally the same code path twice) if left as-is. Those now instead
compare the C++ port's output against a golden fixture captured from the
real pure-Python implementation *before* it was deleted -- see
``_GOLDEN_*`` below (base64-encoded serialized ``ModelProto`` bytes,
inlined directly rather than as a checked-in ``.onnx`` file: this repo's
own ``.gitignore`` excludes ``*.onnx`` outright, and there is no existing
``tests/golden/``-style fixture-directory convention to follow instead --
see CLAUDE.md's own "Prefer onnx.parser-based model construction in tests"
note for the same "keep fixtures in the test file itself" spirit) --
preserving the original regression coverage (did the behavior change?)
without asserting a tautology.
"""

import base64

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _golden(b64):
    return onnx.load_from_string(base64.b64decode(b64))


# Frozen from onnxsim.apply_transformer_block_pruning's own real pure-Python
# implementation, on the exact model + calibration seed each corresponding
# test below builds, before that implementation was deleted in favor of the
# C++ port (see this file's own module docstring).
_GOLDEN_DROPS_NEAR_IDENTITY_MLP_BLOCK = (
    "CAo6qQcKSAoCeDAKCExuMVNjYWxlCgdMbjFCaWFzEgNsbjEiEkxheWVyTm9ybWFsaXphdGlvbioU"
    "CgRheGlzGP///////////wGgAQI6AAoXCgNsbjEKAlcxEgJoMSIGTWF0TXVsOgAKEwoCeDAKAmgx"
    "EgJ4MiIDQWRkOgAKEwoCeDISAXkiCElkZW50aXR5OgASAWcqMAgIEAFCCExuMFNjYWxlSiAAAIA/"
    "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPyovCAgQAUIHTG4wQmlhc0ogAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAqjQIICAgIEAFCAlcwSoACdgAHNLXYDbSC6Ss1c0XhM+vKD7U/"
    "IcI0GgWvNek6fjVR6Dy1jdeptRhPJ7VlfjEzuQcctv/sarQtOae1CJFEtTEZErX2z6m0Ev7cNH3s"
    "izVXAwq0VGe3NdSPMrUtt7w0AIZyNc/jyTPhlEe1fGx3tVy99bTGbmw0N4KHtb+ZYLRw9yq0oC4R"
    "NQ19ZjQOyr40w4Ivte0rC7RiclI153HINSr9qLUGMss17aO0NVC7UTWB+o00OYmotDaxwzX7jAM2"
    "tM/xNZ+CsDX93b80gy2itfgKmbGdODA1w+ustSoh1DQEyOY0sdc6NfztnrXbnzG1NE/qtBYCnbU6"
    "dOk1uh4FtSowCAgQAUIITG4xU2NhbGVKIAAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
    "Ki8ICBABQgdMbjFCaWFzSiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqNAggICAgQ"
    "AUICVzFKgAK1bqg+n2OEvj2vyj+XAak/ZiMiP04GDcBVHFU9DwYvP9CBgD8oLx6/qzfpP+IDqb/n"
    "WSm/cF9vP37tSD0zJwBALQtBPgIZIr8BUMG+raqLvwaLo7+mYiE/SMcUPxq0pT/YLUG/rDTYP3sk"
    "k742hsk/G5bdvqJIPL+9x38+qAaEP7LfJD445RW/Fq2rvwRls7/TrwA/1V19P9c8KL7KhIm/sX9f"
    "P/Pjo7+hiza/B/seP1ACEMA80sU+auYUvwnO3z1nCZu9E/dOPkG1MT+FJEK/veK1P0jhOT/d/lc/"
    "QxqVP2KfST+KFVg/09CaPYegtr9DSQq+6f5Ev2cctr/tU4Q+WhgKAngwEhIKEAgBEgwKAggBCgII"
    "BAoCCAhiFwoBeRISChAIARIMCgIIAQoCCAQKAggIQgQKABAV"
)


_GOLDEN_SPARSITY_SELECTS_FRACTION = (
    "CAo6yggKQgoCeDAKBVNjYWxlCgRCaWFzEgNsbjEiEkxheWVyTm9ybWFsaXphdGlvbioUCgRheGlz"
    "GP///////////wGgAQI6AAoXCgNsbjEKAlcxEgJoMSIGTWF0TXVsOgAKEwoCeDAKAmgxEgJ4MiID"
    "QWRkOgAKEwoCeDISAXkiCElkZW50aXR5OgASAWcqLQgIEAFCBVNjYWxlSiAAAIA/AACAPwAAgD8A"
    "AIA/AACAPwAAgD8AAIA/AACAPyosCAgQAUIEQmlhc0ogAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAqjQIICAgIEAFCAlcwSoACufYINv6BK7YRd+A01WgYtaUD87Tmfme0GY8HthYJebT9"
    "QGi1sgBfNsBvcjQ1Ub202QOXtMxTM7Wyno21R8/RtAlfATWPEoC0V4yANTGJVrRZY9AyAHrPNV5T"
    "EjUOnwe1ZVJEtJsYETWP3AE2VsCQtHPCgrQ8h4Y1EPVttbmdnLSe52w1WckbNdeHxDM44TM1dcs9"
    "ttYTiTUlzYC1V/XftWpqlDQUDTw1YMjutAZ5kLUeaeAyR4xis/ynvDV8oUg1pBtQNG0zlTW/rVy0"
    "Vot4tS7IHDW8Xxw1uatmtDIiUrVZDXY0x1wntgNBOTWS5gM1tPbbtXPBgzM2ZoG11UNLNSqNAggI"
    "CAgQAUICVzFKgALMLwLAUBxqvwmnNT/zBJQ/wxwKwBT//r5A8qc+l/Ubvxmayz8eepi/NoW1Picy"
    "hr+Y9rM/612xvKCXvr587du/D0bXPxm2QD+P6UA/GKaRP9PNsj6qpSO/nNxMv+jZTL+IXq8/xe26"
    "v6yrGL8VeqS+iAJmPhlKEz9p4p+/FXHdv0SlkLsPVps/j85BP4/TXD45YqK+uiKWPtUseb5zNFE/"
    "52BLvzJ2CT6y4OK9mhkLP6QHZj7EMyNA69O/PxWVvz87hwLA+z2uvujNG79xYAg/ktsRwPlVlj/p"
    "kog/QqqmvyiCer+cGU2/C1cxPa4WJD+REANAIy9Kvgx7RD/ZJR8+Ko0CCAgICBABQgJXMkqAAps2"
    "7DWzOEc1Ca+3NaOkkLW5ak60BHJatZn9yTWsiDA1u9KjtLbq8rTrGQI1aE48tZXNebXuMAE1SUwl"
    "NlckhLTONhW1pDCdtaEus7WT7Aw1vWJkNdmeHTLljLI00O34MyfhFDRt1sy1TPP1tGZm7zPQOlK1"
    "Acj/tIPhW7VvC7O0JwFlNfpH2rSnNyW0TW5aNUYULTXjhuM1Q0oMtuYCZjVPdgG1zp8QNKPfYDVM"
    "ZJE10X+LNWeLJjSXC9g1r+uXtNBgF7T/klY1BgIUtRYEETasy4g1GQASNj5m5LJKq820zl0zNLYv"
    "RTW5rx210NnLNABZkLJn2tg1huQxtdhpjDVaGAoCeDASEgoQCAESDAoCCAEKAggECgIICGIXCgF5"
    "EhIKEAgBEgwKAggBCgIIBAoCCAhCBAoAEBU="
)


_GOLDEN_NUM_BLOCKS_TO_DROP_CAPS = (
    "CAo67AUKNwoCeDASAXkaJHkvdHJhbnNmb3JtZXJfYmxvY2tfcHJ1bmluZ19pZGVudGl0eSIISWRl"
    "bnRpdHkSAWcqLQgIEAFCBVNjYWxlSiAAAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPyos"
    "CAgQAUIEQmlhc0ogAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqjQIICAgIEAFCAlcw"
    "SoACzDnUtDCwjTSJ+SI1KHuCtXwRTjXs7og0+C1SNeFAkjRM9ps1Ibp7tVJi7jXNYaE1ygwhtUk2"
    "MTVdyO40+0/qtfPGHzVSJB21mTmGtK6zIbVr0+W0kgyaM+ezzzMNRNG1rUKQtFx/tLWYj6q18US6"
    "tMqXZTUhWCk1yU8itb+zPrUtG161IkQZNBpuezUODJsykPY5NX5VqTRt48k1r6oGtuMCD7aoElW0"
    "qt4qNXdp3LTaluQ0sYKdtRQIqbW4WkC1QtdhNWLtOjVadj+1t3lCNLvULbX9T6wy746VtQzzR7Og"
    "gEW1y4U+swcIL7WllVy1g5fzMz5atbRrJ7oyof7gtCqNAggICAgQAUICVzFKgAIXwCQ1bvVatStc"
    "TjVKxrI1pVuQtBg+6jXzNQg2xN/tNPU2trXS2ti1rb08tHxc7jXEOho1fXtLNUO1hrRiwUe1c6zH"
    "NFnRlLWpd1O0HjYJNQVWGjPIzww1jm7vtJuSIDW6Ptk0emvdteLji7XqNEK1jPsvNZUtojSd7Zc0"
    "Tq85NFTaKjXPw301zucUtVOH1bUwIjM1dt9gtBSeUTM6nzq0jlOVtZXByrVqbIE1YLaSNU36eTUb"
    "l180NymQNEHp4LQfDpIzX0XYtNTxEbXRVVw0SerjNf7N/LWLdyE1Z141Na3+orU7SKg0bSHaM00Z"
    "+bQmOwW1vru8tS8i5TXnih02WhgKAngwEhIKEAgBEgwKAggBCgIIBAoCCAhiFwoBeRISChAIARIM"
    "CgIIAQoCCAQKAggIQgQKABAV"
)


_GOLDEN_INTERIOR_OVERLAP_SKIP = (
    "CAo6xAcKRAoCeDIKBlNjYWxlQgoFQmlhc0ISA2xuQiISTGF5ZXJOb3JtYWxpemF0aW9uKhQKBGF4"
    "aXMY////////////AaABAjoAChkKA2xuQgoDV0IyEgNoQjIiBk1hdE11bDoAChkKA2hCMgoCeDAS"
    "B2hCRmluYWwiA0FkZDoAChcKAngyCgdoQkZpbmFsEgF5IgNBZGQ6ABIBZyouCAgQAUIGU2NhbGUw"
    "SiAAAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPyotCAgQAUIFQmlhczBKIAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKo0CCAgICBABQgJXQUqAAufvjj0JqLG9nf/rvINagD1r"
    "+0Y+bdwNvsVf671rfRU9tdz+vRPO/71sZ609+gVePIt8or1L0Lo86WkAvX+skb21Q3C+W7eSPcuW"
    "l72RNrG8hPsHvrq6Vj0ueMM94gIjPi5qOT2lh949I365vNyRBb6wOho+n631Ojb8nD2NRTW8HanY"
    "PafXPj59YE492qD3PTEri7xnQBM+r39pPORctT06meS9JwExPctvxj3zkwA995B2vkHBm7yjDA8+"
    "/mehvZtgFL2isYG9ZWphPUgo1r2wka+9ul6NPcFDib6+Qhe+GKQXvh+a2LvrVMC92hbZvRudf74j"
    "Guu9G6NxvU6eLr4qLggIEAFCBlNjYWxlQkogAACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAA"
    "gD8qLQgIEAFCBUJpYXNCSiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACqOAggICAgQ"
    "AUIDV0IySoACN1eEu3MfOT5PPDM+CJOSOtb8iry+L6G9dSlavOIE6L3Vfck7KDVNvNCuZb1HfZu9"
    "Qx8OvI4DAr1GxcS9HZnJPcfTDT4HBH47cxqQvVMFWrupNaG8715NvaTvOL4wV809pmoSPsh43jyg"
    "1ke9P/oFvUuZtj18zLq9INXeva07zr3TxI089t0wvF/Sbbu6yxg+LAs+Pd7fk7wde328ZdZjPuD8"
    "1j1Tc0Q+doUnvBm9Mz7CQ9c9oHd8vWcQFr34aTC+OntKPOK7Wj2A7le9ozYmOyg0OD47Rr49pTmZ"
    "PWC/D75sQBE9LYdgviKUBD6IlCS9mF8nu9uhSL0VsCS6vS1WvFoYCgJ4MBISChAIARIMCgIIAQoC"
    "CAQKAggIWhgKAngyEhIKEAgBEgwKAggBCgIIBAoCCAhiFwoBeRISChAIARIMCgIIAQoCCAQKAggI"
    "QgQKABAV"
)


_GOLDEN_MATCHES_RANDOM_MODEL = (
    "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"
)


def _model(body, initializer=(), opset=21):
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


# --- Basic drop: matches a hand-built "already pruned" oracle ---------------


def test_transformer_block_pruning_cpp_drops_near_identity_mlp_block_and_matches_manual_removal_oracle():
    # Two stacked pre-norm MLP residual blocks, block 0 engineered
    # near-identity (tiny down-projection weight) -- the canonical
    # "redundant block" case the calibrated mean-cosine-similarity ranking
    # should flag as the one to drop. Mirrors
    # test_pruning.py's own
    # test_transformer_block_pruning_drops_near_identity_mlp_block_and_matches_manual_removal_oracle
    # exactly, just through the C++-backed entry point.
    H = 8
    rng = np.random.default_rng(0)
    ln0_scale = np.ones(H, dtype=np.float32)
    ln0_bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    ln1_scale = np.ones(H, dtype=np.float32)
    ln1_bias = np.zeros(H, dtype=np.float32)
    w1 = rng.standard_normal((H, H)).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Ln0Scale, Ln0Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Ln1Scale, Ln1Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x1, h1)

          y = Identity(x2)
        }}
        """,
        initializer=[
            _f32(ln0_scale, "Ln0Scale"),
            _f32(ln0_bias, "Ln0Bias"),
            _f32(w0, "W0"),
            _f32(ln1_scale, "Ln1Scale"),
            _f32(ln1_bias, "Ln1Bias"),
            _f32(w1, "W1"),
        ],
    )
    onnx.checker.check_model(model)

    ref = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln1 = LayerNormalization<axis=-1>(x0, Ln1Scale, Ln1Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x0, h1)
          y = Identity(x2)
        }}
        """,
        initializer=[
            _f32(ln1_scale, "Ln1Scale"),
            _f32(ln1_bias, "Ln1Bias"),
            _f32(w1, "W1"),
        ],
    )
    onnx.checker.check_model(ref)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)

    assert [n.op_type for n in pruned.graph.node] == [
        "LayerNormalization",
        "MatMul",
        "Add",
        "Identity",
    ]

    x = np.random.default_rng(42).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    (ref_y,) = _run(ref, {"x0": x})
    np.testing.assert_array_equal(pruned_y, ref_y)

    # Cross-check against a golden fixture frozen from the pure-Python
    # reference on the identical model + calibration seed, back when
    # apply_transformer_block_pruning still had its own implementation (see
    # this file's own module docstring) -- the primary port-correctness
    # signal.
    golden = _golden(_GOLDEN_DROPS_NEAR_IDENTITY_MLP_BLOCK)
    assert pruned.SerializeToString() == golden.SerializeToString()


def test_transformer_block_pruning_cpp_adversarial_ranking_prefers_more_redundant_block_regardless_of_position():
    # The redundant block is placed *second*, proving selection follows the
    # calibrated cosine-similarity ranking, not simply "whichever candidate
    # is found first".
    H = 8
    rng = np.random.default_rng(7)
    ln0_scale = np.ones(H, dtype=np.float32)
    ln0_bias = np.zeros(H, dtype=np.float32)
    w0 = rng.standard_normal((H, H)).astype(np.float32)
    ln1_scale = np.ones(H, dtype=np.float32)
    ln1_bias = np.zeros(H, dtype=np.float32)
    w1 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Ln0Scale, Ln0Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Ln1Scale, Ln1Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x1, h1)

          y = Identity(x2)
        }}
        """,
        initializer=[
            _f32(ln0_scale, "Ln0Scale"),
            _f32(ln0_bias, "Ln0Bias"),
            _f32(w0, "W0"),
            _f32(ln1_scale, "Ln1Scale"),
            _f32(ln1_bias, "Ln1Bias"),
            _f32(w1, "W1"),
        ],
    )

    ref = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Ln0Scale, Ln0Bias)
          h0 = MatMul(ln0, W0)
          y = Add(x0, h0)
        }}
        """,
        initializer=[
            _f32(ln0_scale, "Ln0Scale"),
            _f32(ln0_bias, "Ln0Bias"),
            _f32(w0, "W0"),
        ],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=1, num_samples=4
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == [
        "LayerNormalization",
        "MatMul",
        "Add",
        "Identity",
    ]
    survivor = next(n for n in pruned.graph.node if n.op_type == "MatMul")
    w_name = survivor.input[1]
    inits = {t.name: t for t in pruned.graph.initializer}
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits[w_name]), w0)

    x = np.random.default_rng(99).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    (ref_y,) = _run(ref, {"x0": x})
    np.testing.assert_array_equal(pruned_y, ref_y)


def test_transformer_block_pruning_cpp_drops_attention_block_and_matches_manual_removal_oracle():
    # A single self-attention residual block (Q/K/V feeding a real
    # com.microsoft::GroupQueryAttention node, not a plain MLP) -- confirms
    # F need not be an MLP at all.
    K, NH, D = 16, 2, 8
    Nqkv = NH * D
    rng = np.random.default_rng(5)
    scale = np.ones(K, dtype=np.float32)
    bias = np.zeros(K, dtype=np.float32)
    wq = (rng.standard_normal((K, Nqkv)) * 1e-6).astype(np.float32)
    wk = (rng.standard_normal((K, Nqkv)) * 1e-6).astype(np.float32)
    wv = (rng.standard_normal((K, Nqkv)) * 1e-6).astype(np.float32)
    wout = (rng.standard_normal((Nqkv, K)) * 1e-6).astype(np.float32)
    seq = 5
    seqlens_k = np.full((2,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)

    model = _model(
        f"""
        g (float[2,{seq},{K}] x0) => (float[2,{seq},{K}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          q = MatMul(ln, Wq)
          k = MatMul(ln, Wk)
          v = MatMul(ln, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={NH}, kv_num_heads={NH}> (q, k, v, , , SeqLensK, TotalSeq)
          fout = MatMul(ctx, Wout)
          y = Add(x0, fout)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
            onnx.numpy_helper.from_array(seqlens_k, "SeqLensK"),
            onnx.numpy_helper.from_array(total_seq, "TotalSeq"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)

    assert [n.op_type for n in pruned.graph.node] == ["Identity"]
    assert list(pruned.graph.node[0].input) == ["x0"]
    assert list(pruned.graph.node[0].output) == ["y"]

    x = np.random.default_rng(17).standard_normal((2, 5, K)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    np.testing.assert_array_equal(pruned_y, x)


# --- Fused SkipLayerNormalization entry norm ---------------------------------


def test_transformer_block_pruning_cpp_matches_fused_skip_layer_normalization_entry():
    # A fused com.microsoft::SkipLayerNormalization node as the block's own
    # *entry* norm -- exactly what onnxruntime's transformer optimizer
    # produces by fusing the *previous* residual Add with the following
    # LayerNormalization. This block's own merge itself stays an ordinary,
    # unfused Add.
    H = 8
    rng = np.random.default_rng(41)
    gamma = rng.standard_normal(H).astype(np.float32)
    beta = rng.standard_normal(H).astype(np.float32)
    w = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    def _skip_ln_node(sum_output_name):
        return onnx.helper.make_node(
            "SkipLayerNormalization",
            ["input", "skip", "Gamma", "Beta"],
            ["ln_out", "", "", sum_output_name],
            domain="com.microsoft",
            epsilon=1e-5,
        )

    initializer = [_f32(gamma, "Gamma"), _f32(beta, "Beta")]
    inputs = [
        onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, H]),
        onnx.helper.make_tensor_value_info("skip", onnx.TensorProto.FLOAT, [1, 4, H]),
    ]

    skip_ln = _skip_ln_node("sum_out")
    h = onnx.helper.make_node("MatMul", ["ln_out", "W"], ["h"])
    y = onnx.helper.make_node("Add", ["sum_out", "h"], ["y"])
    graph = onnx.helper.make_graph(
        [skip_ln, h, y],
        "g",
        inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, H])],
        initializer=initializer + [_f32(w, "W")],
        # Plain onnx.shape_inference has no inference function for a
        # com.microsoft op, so this declared value_info is what makes
        # ApplyTransformerBlockPruning's own TransformerBlockShapesMatch
        # check actually confirm the match (see test_pruning.py's own
        # identical test for the full reasoning).
        value_info=[
            onnx.helper.make_tensor_value_info(
                "sum_out", onnx.TensorProto.FLOAT, [1, 4, H]
            )
        ],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model)

    ref_skip_ln = _skip_ln_node("y")
    ref_graph = onnx.helper.make_graph(
        [ref_skip_ln],
        "g",
        inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, H])],
        initializer=initializer,
    )
    ref = onnx.helper.make_model(
        ref_graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(ref)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)

    # The fused SkipLayerNormalization node itself survives, unchanged --
    # deleting it would delete the very tensor (sum_out) every rewired
    # x_out consumer now reads.
    assert [n.op_type for n in pruned.graph.node] == [
        "SkipLayerNormalization",
        "Identity",
    ]

    x_in = np.random.default_rng(42).standard_normal((1, 4, H)).astype(np.float32)
    x_skip = np.random.default_rng(43).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"input": x_in, "skip": x_skip})
    (ref_y,) = _run(ref, {"input": x_in, "skip": x_skip})
    np.testing.assert_array_equal(pruned_y, ref_y)


# --- sparsity vs num_blocks_to_drop sizing -----------------------------------


def test_transformer_block_pruning_cpp_sparsity_selects_fraction_of_matched_candidates():
    # Three stacked blocks, 0 and 2 engineered near-identity, block 1 a real
    # transform. sparsity=2/3 of 3 matched candidates rounds to 2 -- both
    # near-identity blocks should be dropped, leaving only block 1, rewired
    # straight from x0. Also exercises two NON-adjacent committed drops (0
    # and 2, sandwiching a kept block): block 2's own x_in is itself
    # downstream of block 0's own now-deleted merge, not a graph input --
    # CommitTransformerBlockDrops' own `resolve` chaining must handle this.
    H = 8
    rng = np.random.default_rng(3)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    w1 = rng.standard_normal((H, H)).astype(np.float32)
    w2 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Scale, Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x1, h1)

          ln2 = LayerNormalization<axis=-1>(x2, Scale, Bias)
          h2 = MatMul(ln2, W2)
          x3 = Add(x2, h2)

          y = Identity(x3)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w0, "W0"),
            _f32(w1, "W1"),
            _f32(w2, "W2"),
        ],
    )

    ref = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln1 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h1 = MatMul(ln1, W1)
          y = Add(x0, h1)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w1, "W1")],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, sparsity=2 / 3, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node].count("LayerNormalization") == 1

    x = np.random.default_rng(11).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    (ref_y,) = _run(ref, {"x0": x})
    np.testing.assert_array_equal(pruned_y, ref_y)

    golden = _golden(_GOLDEN_SPARSITY_SELECTS_FRACTION)
    assert pruned.SerializeToString() == golden.SerializeToString()


def test_transformer_block_pruning_cpp_num_blocks_to_drop_caps_at_matched_candidate_count():
    # Only 2 candidates exist (both engineered near-identity);
    # num_blocks_to_drop=10 is silently capped rather than erroring -- both
    # get dropped, and the model collapses to a straight identity.
    H = 8
    rng = np.random.default_rng(31)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    w1 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Scale, Bias)
          h1 = MatMul(ln1, W1)
          y = Add(x1, h1)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w0, "W0"),
            _f32(w1, "W1"),
        ],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=10, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Identity"]

    x = np.random.default_rng(33).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    np.testing.assert_array_equal(pruned_y, x)

    golden = _golden(_GOLDEN_NUM_BLOCKS_TO_DROP_CAPS)
    assert pruned.SerializeToString() == golden.SerializeToString()


def test_transformer_block_pruning_cpp_num_blocks_to_drop_takes_priority_over_sparsity():
    # Both given: num_blocks_to_drop wins (mirrors pruning.py's own keyword
    # precedence -- see apply_transformer_block_pruning's own signature).
    H = 8
    rng = np.random.default_rng(200)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    w1 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Scale, Bias)
          h1 = MatMul(ln1, W1)
          y = Add(x1, h1)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w0, "W0"),
            _f32(w1, "W1"),
        ],
    )

    # sparsity=0.0 alone would drop nothing; num_blocks_to_drop=1 overrides.
    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, sparsity=0.0, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert [n.op_type for n in pruned.graph.node].count("LayerNormalization") == 1
    onnx.checker.check_model(pruned)


# --- Decline cases: left completely untouched --------------------------------


def test_transformer_block_pruning_cpp_declines_fused_entry_when_sum_output_absent():
    H = 8
    rng = np.random.default_rng(49)
    gamma = rng.standard_normal(H).astype(np.float32)
    beta = rng.standard_normal(H).astype(np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)

    skip_ln = onnx.helper.make_node(
        "SkipLayerNormalization",
        ["input", "skip", "Gamma", "Beta"],
        ["ln_out"],  # no fourth output declared at all
        domain="com.microsoft",
        epsilon=1e-5,
    )
    h = onnx.helper.make_node("MatMul", ["ln_out", "W"], ["h"])
    y = onnx.helper.make_node("Add", ["input", "h"], ["y"])
    graph = onnx.helper.make_graph(
        [skip_ln, h, y],
        "g",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, 4, H]
            ),
            onnx.helper.make_tensor_value_info(
                "skip", onnx.TensorProto.FLOAT, [1, 4, H]
            ),
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, H])],
        initializer=[_f32(gamma, "Gamma"), _f32(beta, "Beta"), _f32(w, "W")],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_kv_cache_bearing_attention_block():
    # present_k/present_v are declared as graph outputs -- the generic "no
    # block-internal node's own output may leak outside the block" check
    # catches this, with no KV-cache-specific detection at all.
    K, NH, D = 8, 2, 4
    Nqkv = NH * D
    rng = np.random.default_rng(19)
    scale = np.ones(K, dtype=np.float32)
    bias = np.zeros(K, dtype=np.float32)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, K)).astype(np.float32)

    model = _model(
        f"""
        g (float[2,5,{K}] x0) => (float[2,5,{K}] y, float[2,5,{Nqkv}] present_k, float[2,5,{Nqkv}] present_v)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          q = MatMul(ln, Wq)
          k = MatMul(ln, Wk)
          v = MatMul(ln, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={NH}, kv_num_heads={NH}> (q, k, v)
          fout = MatMul(ctx, Wout)
          y = Add(x0, fout)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_when_intermediate_tensor_has_external_consumer():
    H = 8
    rng = np.random.default_rng(15)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y, float[1,4,{H}] ln_out)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
          ln_out = Identity(ln)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_shape_broadcasting_merge():
    # F's own final output is tiled to a *wider* batch dimension than
    # x_in's own -- the residual Add would silently broadcast x_in up to
    # match, so replacing every x_out consumer with (narrower) x_in
    # directly would be shape-unsafe. Declined via TransformerBlockShapesMatch,
    # not guessed at -- this exercises the real onnx::shape_inference path
    # (not just whatever value_info the input model happens to carry).
    H = 8
    rng = np.random.default_rng(21)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    repeats = np.array([3, 1, 1], dtype=np.int64)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[3,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          h_tiled = Tile(h, Repeats)
          y = Add(x0, h_tiled)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w, "W"),
            onnx.numpy_helper.from_array(repeats, "Repeats"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_when_f_reads_x_in_directly():
    # No LayerNorm/RMSNorm at all -- F reads x0 raw. No candidate is even
    # found.
    H = 8
    rng = np.random.default_rng(25)
    w = rng.standard_normal((H, H)).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          h = MatMul(x0, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_no_candidates_returns_unchanged_copy():
    H = 8
    rng = np.random.default_rng(35)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          y = MatMul(x0, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(model, num_blocks_to_drop=1)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_zero_sparsity_leaves_model_untouched():
    H = 8
    rng = np.random.default_rng(37)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- Interior-overlap-skip: two independently-matched candidates whose own
# block_nodes overlap -----------------------------------------------------


def test_transformer_block_pruning_cpp_interior_overlap_skip_matches_python_reference():
    # Constructs a genuine "candidate B's own interior fully contains
    # candidate A's own interior" topology -- the unusual-but-possible case
    # SelectDroppableBlocks/_select_droppable_blocks' own docstring
    # describes, where only ONE of two independently-VALID candidates can
    # ever safely be committed:
    #
    #   Block A: x1 = x0 + MatMul(LN0(x0), WA)               (candidate A)
    #   Block B: y  = x2 + (MatMul(LN_B(x2), WB2) + x1)      (candidate B)
    #
    # Block B's own F (its "other" merge operand) is built by summing a
    # normal LN_B(x2)-derived term with x1 -- Block A's own raw output --
    # used as a plain additive term, not through any norm. Walking B's own
    # backward search from its own merge therefore passes straight through
    # Add_A (the node that produces x1) and everything upstream of it
    # (MatMul_A, LN0), collecting them as ordinary interior nodes of B, in
    # ADDITION to B's own genuine LN_B(x2) boundary elsewhere in the sum.
    # Since x1's own only real consumer is that one sum node inside F_B (no
    # other reader, no graph-output exposure), and x_out (a merge node's own
    # primary output) is exempt from the "no external consumer" check for
    # ITS OWN candidate, BOTH candidate A (interior {Add_A, MatMul_A, LN0})
    # and candidate B (interior a strict superset: {Add_B, the sum node,
    # MatMul_B2, LN_B, Add_A, MatMul_A, LN0}) independently pass every
    # safety check and are both matched -- with block_nodes(A) subset-of
    # block_nodes(B).
    #
    # num_blocks_to_drop=2 (both "found" candidates) can therefore never
    # actually commit two independent drops: whichever candidate the
    # ranking tries first gets committed, and the second is SKIPPED outright
    # (its own block_nodes overlaps the first commit's), never causing the
    # whole call to decline. The primary assertion here is exact parity
    # with the pure-Python reference on the identical model + calibration
    # data -- proving the C++ port makes the identical skip decision the
    # Python original does, whichever way the (calibration-dependent)
    # ranking happens to resolve.
    H = 8
    rng = np.random.default_rng(500)
    scale0 = np.ones(H, dtype=np.float32)
    bias0 = np.zeros(H, dtype=np.float32)
    wa = rng.standard_normal((H, H)).astype(np.float32) * 0.1
    scale_b = np.ones(H, dtype=np.float32)
    bias_b = np.zeros(H, dtype=np.float32)
    wb2 = rng.standard_normal((H, H)).astype(np.float32) * 0.1

    model = _model(
        f"""
        g (float[1,4,{H}] x0, float[1,4,{H}] x2) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale0, Bias0)
          hA = MatMul(ln0, WA)
          x1 = Add(x0, hA)

          lnB = LayerNormalization<axis=-1>(x2, ScaleB, BiasB)
          hB2 = MatMul(lnB, WB2)
          hBFinal = Add(hB2, x1)
          y = Add(x2, hBFinal)
        }}
        """,
        initializer=[
            _f32(scale0, "Scale0"),
            _f32(bias0, "Bias0"),
            _f32(wa, "WA"),
            _f32(scale_b, "ScaleB"),
            _f32(bias_b, "BiasB"),
            _f32(wb2, "WB2"),
        ],
    )
    onnx.checker.check_model(model)

    rng_cal = np.random.default_rng(501)
    calibration_data = [
        {
            "x0": rng_cal.standard_normal((1, 4, H)).astype(np.float32),
            "x2": rng_cal.standard_normal((1, 4, H)).astype(np.float32),
        }
        for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_transformer_block_pruning_cpp(
        model, calibration_data=calibration_data, num_blocks_to_drop=2
    )
    onnx.checker.check_model(pruned_cpp)
    golden = _golden(_GOLDEN_INTERIOR_OVERLAP_SKIP)
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()

    # Confirm the graph really did change (at least one candidate was
    # committed) -- this isn't a "nothing matched" no-op.
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    # And confirm the result still executes correctly (the rewiring/
    # deletion mechanics produced a valid, runnable graph even in this
    # overlap-heavy topology).
    x0 = np.random.default_rng(502).standard_normal((1, 4, H)).astype(np.float32)
    x2 = np.random.default_rng(503).standard_normal((1, 4, H)).astype(np.float32)
    _run(pruned_cpp, {"x0": x0, "x2": x2})  # must not raise


# --- Cross-check against the pure-Python reference on random multi-block
# models -----------------------------------------------------------------


def test_transformer_block_pruning_cpp_matches_python_reference_random_model():
    H = 12
    rng = np.random.default_rng(600)
    scales = [np.ones(H, dtype=np.float32) for _ in range(4)]
    biases = [np.zeros(H, dtype=np.float32) for _ in range(4)]
    # A mix of near-identity and real-transform blocks so the ranking is
    # meaningfully exercised.
    weights = [
        (rng.standard_normal((H, H)) * (1e-6 if i % 2 == 0 else 1.0)).astype(np.float32)
        for i in range(4)
    ]

    body_lines = []
    prev = "x0"
    for i in range(4):
        body_lines.append(
            f"ln{i} = LayerNormalization<axis=-1>({prev}, Scale{i}, Bias{i})"
        )
        body_lines.append(f"h{i} = MatMul(ln{i}, W{i})")
        nxt = "y" if i == 3 else f"x{i + 1}"
        body_lines.append(f"{nxt} = Add({prev}, h{i})")
        prev = nxt
    body = "\n".join(body_lines)

    initializer = []
    for i in range(4):
        initializer.append(_f32(scales[i], f"Scale{i}"))
        initializer.append(_f32(biases[i], f"Bias{i}"))
        initializer.append(_f32(weights[i], f"W{i}"))

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
        {body}
        }}
        """,
        initializer=initializer,
    )
    onnx.checker.check_model(model)

    rng_cal = np.random.default_rng(601)
    calibration_data = [
        {"x0": rng_cal.standard_normal((1, 4, H)).astype(np.float32)} for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_transformer_block_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    golden = _golden(_GOLDEN_MATCHES_RANDOM_MODEL)
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()
    # Sanity: some real reduction happened.
    assert len(pruned_cpp.graph.node) < len(model.graph.node)


# --- Error handling ------------------------------------------------------


def test_transformer_block_pruning_cpp_missing_calibration_input_raises():
    H = 8
    rng = np.random.default_rng(700)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    bad_batch = {"NotX0": np.zeros((1, 4, H), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_transformer_block_pruning_cpp(
            model, calibration_data=[bad_batch], num_blocks_to_drop=1
        )


def test_transformer_block_pruning_cpp_negative_num_blocks_to_drop_raises():
    H = 8
    rng = np.random.default_rng(701)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    with pytest.raises(Exception):
        onnxsim.apply_transformer_block_pruning_cpp(model, num_blocks_to_drop=-1)


def test_transformer_block_pruning_cpp_sparsity_out_of_range_raises():
    H = 8
    rng = np.random.default_rng(702)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    with pytest.raises(Exception):
        onnxsim.apply_transformer_block_pruning_cpp(model, sparsity=1.5)


# --- Default (auto-generated) calibration data -------------------------------


def test_transformer_block_pruning_cpp_default_calibration_data_runs():
    H = 8
    rng = np.random.default_rng(800)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_samples=4, seed=5, num_blocks_to_drop=1
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Identity"]
