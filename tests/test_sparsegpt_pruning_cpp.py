"""Tests for ``onnxsim.apply_sparsegpt_pruning_cpp`` -- the C++-backed port
of ``onnxsim.apply_sparsegpt_pruning`` (SparseGPT, Frantar & Alistarh, 2023;
see ``onnxsim/structured_pruning_entry.cpp``'s own "SparseGPT (unstructured /
N:M) pruning" section and ``ApplySparseGptPruning``). Like
``test_structured_wanda_pruning_cpp.py``, this runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor.

Unlike every other C++-ported pruning pass tested elsewhere in this test
suite (all purely structural: they drop whole rows/columns and leave every
surviving entry byte-for-byte unchanged), SparseGPT RECOMPUTES every
surviving entry's own value via a sequential, Hessian-error-compensating
update -- so "reaches the target sparsity" is nowhere near enough to prove
this port correct. The tests below instead compare this port's actual
output, entry for entry, against the pure-Python ``onnxsim.apply_sparsegpt_
pruning`` reference (same calibration data, same parameters) -- both
implementations solve for the mathematically unique Cholesky factor of the
same damped Hessian, so a correct port should match to (and, empirically,
essentially at) floating-point precision, not merely a loose tolerance.

Scope: this port matches plain ``MatMul``/vanilla-``Gemm`` (not
``com.microsoft::FusedGemm``/``GemmFastGelu``), ``com.microsoft::
Attention``'s merged QKV weight, FLOAT32/FLOAT16/BFLOAT16 (widened from an
earlier FLOAT32-only scope -- see ``IsSupportedFloatDtype``/
``ReadTensorAsF64``/``WriteF64TensorAs`` and, for the Attention QKV weight
specifically, the SparseGPT-local ``MatchAttentionProducerAnyFloat``), and
-- as of this round -- every 2-D ``Conv``/``FusedConv`` node (ordinary/
depthwise/general-grouped alike), closing what was previously the one
remaining scope gap versus the pure-Python original. See
``structured_pruning_entry.h``'s own ``ApplySparseGptPruning`` declaration
comment for the full scope, the Conv im2col cross-covariance Hessian this
needed, and why pruning.py's own Conv implementation -- despite having no
correct upstream SparseGPT reference of its own to port from or check
against -- was independently re-verified this round as trustworthy ground
truth to port and check this C++ port's own Conv machinery against
numerically (an independent nested-loop-oracle Hessian and a from-scratch
``fasterprune`` transliteration, both already established in
``tests/test_pruning.py``).

``onnxsim.apply_sparsegpt_pruning`` (the pure-Python name) is now itself a
thin alias for :func:`onnxsim.apply_sparsegpt_pruning_cpp` (full parity
verified -- see pruning.py's own docstring there), so the tests below that
used to call BOTH entry points and compare their live outputs would be
tautological (literally the same code path twice) if left as-is. Those now
instead compare this port's output against a golden fixture captured from
the real pure-Python implementation *before* it was aliased away -- see
``_GOLDEN_*`` below (base64-encoded serialized ``ModelProto`` bytes, inlined
directly rather than as a checked-in ``.onnx`` file -- see
``test_transformer_block_pruning_cpp.py``'s own module docstring for why:
this repo's own ``.gitignore`` excludes ``*.onnx`` outright, and there is no
existing ``tests/golden/``-style fixture-directory convention to follow
instead) -- preserving the original regression coverage (did the behavior
change?) without asserting a tautology. The new Conv tests further down are
NOT frozen this way -- Conv is genuinely new coverage this round, verified
against an independent nested-loop im2col Hessian oracle and a from-scratch
``fasterprune`` transliteration (mirroring ``tests/test_pruning.py``'s own
verification bar for the same Conv machinery), never against
``onnxsim.apply_sparsegpt_pruning`` itself.
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


def _matmul_model(K=32, N=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return (
        _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{N}] Y)
            {{
              Y = MatMul(X, W)
            }}
            """,
            initializer=[_f32(weight, "W")],
        ),
        weight,
    )


def _weight(model, index=0):
    return onnx.numpy_helper.to_array(model.graph.initializer[index])


def _assert_bytewise_close(actual, expected, rtol=1e-5, atol=1e-6):
    # SparseGPT recomputes every kept entry too, so a correct port should
    # match the Python reference at essentially floating-point precision --
    # this module's own docstring explains why a loose tolerance would not
    # actually prove the port correct.
    np.testing.assert_allclose(
        actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
    )


# Frozen from onnxsim.apply_sparsegpt_pruning's own real pure-Python
# implementation, on the exact model + calibration seed each corresponding
# test below builds, before that implementation was aliased away to this
# same C++ port (see this file's own module docstring).
_GOLDEN_MATMUL_UNSTRUCTURED = (
    "CAo62wgKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAgIIAgIEAFCAVdKgAgAAAAAAAAAAAAAAADe"
    "kjG/1TtVPwAAAAAAAAAAAAAAAAAAAAAAAAAAleAsvwAAAAAAAAAAAAAAAKM/rb8Oqki/AAAAAAAA"
    "AAAAAAAAhjcMvwAAAAAAAAAAY0zPv59tBr8zReC+AAAAAAAAAADRcgy/AAAAAAAAAAAAAAAAAAAA"
    "AHemFj95xoA/AAAAAPzP0r4AAAAAgXl0PwAAAAAAAAAAp5OivkuCDL8LZgy/5pa2Pwocr74jWhK/"
    "AAAAACN+sz6H3b++AAAAAAAAAACbwsE+5KL3vgAAAACcYI8/wcK4viIVDL8AAAAAAAAAAAAAAAAA"
    "AAAAAAAAAPLVBz4AAAAA2EHEPixGmL8AAAAAf2HVvuk5+j4AAAAAuNdPPwAAAAAAAAAAAAAAAHyA"
    "tT4AAAAA7QXxvtpO3r6snUI+lBtSP5OE/77vkag+Ek5OvwaA7T4AAAAAAAAAAKmbBj8GZm8/AAAA"
    "AFmi+L5E9RC/AAAAAAAAAAAAAAAAdAkavwAAAAAAAAAA8cQTPwAAAABpwEI/cYsCP0k9CD8AAAAA"
    "yg1IvwAAAABz5QU/AAAAALfLFb8AAAAAAAAAAANlDL8I7lM/AAAAAD2uIz/yUuu+AAAAAAAAAADH"
    "VNg+DITovgAAAADPu+K+zCA7v/sPBL8AAAAAAAAAAAAAAAAiqMi+vsj8PgAAAAAAAAAAAAAAAAAA"
    "AACn0Ac/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2WfwvgAAAAAAAAAACLw3P5a4EL8AAAAAyTw4"
    "P2QJnj+w346/AAAAAG7pSr8AAAAAAAAAALH9pj5x8ZQ/AAAAAAAAAAAKPOo+7bEnPwAAAABozyw/"
    "L+sgvwAAAAAAAAAApLzCPgAAAAAAAAAAd9R3P6ILCr8AAAAAAAAAADmcx7+fiSy/AAAAAAAAAADH"
    "z1G/AAAAAAAAAAAAAAAAAAAAAAAAAABntSM/0ttKv9lmyT4ZaBQ/AAAAADWdgD8AAAAA5gL3Pph6"
    "vj47JGC/AAAAAAYbIb8AAAAA3HX7vgAAAACF+PC+AAAAAAAAAAAbZCQ/duUAPwAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAEBiA/AAAAAGoACj8AAAAAAAAAAAAAAAAAAAAA2w/cPhzeWz8qtu6+"
    "AAAAAEMr2z4AAAAANBkmv30Agj8AAAAAt928PgAAAACFuq0/duxUvgAAAADR3vk+AAAAAJdjXb8A"
    "AAAApC+6vhp0CD/Hmc++bBEtvwAAAACvWmU/CeqEPwAAAAC/1xQ/AAAAAFrJtr4AAAAAnooTv2ba"
    "pT49My0/AAAAAAAAAABVHBk/UhkDvwAAAAAAAAAAAAAAALXtYj8AAAAAWhgKAVgSEwoRCAESDQoH"
    "EgViYXRjaAoCCCBiGAoBWRITChEIARINCgcSBWJhdGNoCgIICEIECgAQFQ=="
)

_GOLDEN_GEMM_TRANSB = (
    "CAo6zgUKIwoBWAoBVwoBQhIBWSIER2VtbSoNCgZ0cmFuc0IYAaABAjoAEgFnKswECAYIGBABQgFX"
    "SsAEAAAAAAAAAAAAAAAAOhITPwAAAAB0LAW/lm+xvjfjJT8AAAAAX7D4vgAAAAAAAAAAAAAAAMOq"
    "Xj8AAAAAdb4Ov3S2Bz8AAAAAAAAAAHIXc74+i8m+ZsAsP4GiPr8yJI6+3+DqvpfVEz8AAAAAlhOS"
    "vqhgfT7S10I/SVDFvmLLvb4AAAAAAAAAACVOu77UMpq+AAAAAE0LoD5whhC/AAAAAAAAAACvRWA/"
    "rkuAvwAAAAAAAAAAzM0QP63k6L4AAAAAAAAAAAAAAAAAAAAAEYQZv/B2Hj/HUXC+AAAAACxO3D4A"
    "AAAAsMqIPm50Lb8AAAAAAAAAALRZD7+qNLe+iZADv2+73j7ibei+SGPgvgAAAAAAAAAA7QV9P+a/"
    "YD8AAAAAL7aqvrh4dT8AAAAAXAU6vwAAAACf3Tq/uoQNPwAAAACIdv6+X/hwvkddbr+JJC4/AAAA"
    "AAAAAAD3pwE/HoJlvvALlr4AAAAAAAAAAFL0xr4LoG++AAAAAAAAAABvDIy+Qhj6vgAAAACdJps+"
    "Z/UkP0ZhFT8AAAAAAU1vPgAAAABYLTs/Cj/Jvj9o8T4AAAAAAAAAAPUAhz8+R1I/ZrU8vwAAAABm"
    "StQ+0Bi8PjL1AL8AAAAAAAAAAEpX0r5KnLk+AAAAAAAAAAA8svG+Oc5OPuJKkD4AAAAASFPLvgAA"
    "AAAAAAAAAAAAAAAAAABpKeI+AAAAAAAAAAA7tVc/uWjmvgAAAAAAAAAAqHfovs0/Jr+t2L2+n1+A"
    "vwAAAAAAAAAAKiEIBhABQgFCShjT05s924xSPkqkljzGRq888O9ZPaOnX71aGAoBWBITChEIARIN"
    "CgcSBWJhdGNoCgIIGGIYCgFZEhMKEQgBEg0KBxIFYmF0Y2gKAggGQgQKABAV"
)

_GOLDEN_NM_PATTERN = (
    "CAo62wgKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAgIIAgIEAFCAVdKgAgAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAHs3BL8WI4w+7IciviU+AD/52iI/vr9bv3QlJz/0GbS+fhULvwAA"
    "AAAAAAAAimwAP7TctT9hQja/AAAAAH0JRj4AAAAAXMhZv4s/7r0AAAAAAAAAAAAAAAAkoC4/AAAA"
    "ADIKsj4AAAAAfmKovwgcxD4AAAAAaIPbPgAAAAAAAAAAAAAAAGcwAD8AAAAAAAAAAAAAAAAAAAAA"
    "lFiQP6Odd76Wl7A/AAAAAAAAAAAAAAAAOYvPPsp4TD+jYNi+AAAAAAAAAACLuwC/odUgv+HjRT7r"
    "XgA/AAAAAAAAAAAQZqU9WWIWPwAAAAC9pRc/AAAAAOst4D60w58+AAAAAAAAAAAAAAAAfopnPgAA"
    "AAAAAAAAAAAAAAAAAAAAAAAANdcIvwAAAAASqGC/ox7/vo6UqT4AAAAAAAAAABN6Bb8VIz+/OAwl"
    "vwAAAAAAAAAAusNGPwaKgD8x9DU/ypvSvgAAAADTNmi+AAAAAAAAAAAAAAAAAAAAAAAAAADzVXC/"
    "AAAAAF0REz8AAAAAZm4Uv+Cjez4AAAAAc1tRvwAAAAAAAAAAAAAAABREZD8AAAAAnS4WPu4VGT4v"
    "Odk+fDfQvhMUnz4AAAAA8juNP6t+YT8AAAAA5VZGPwAAAAAAAAAAHCZbP24bbL8AAAAAAAAAAKir"
    "l74AAAAAUJwWvwAAAAAAAAAAAAAAAGORrb4AAAAAii6lPtD0u77EOoU/ID4Tvzp2dj8hj4E/61gK"
    "v8pYVD8AAAAAAAAAAAAAAACbzEO/AAAAAAAAAAAAAAAAzmWavwAAAAAJLQE/AAAAAAAAAADrzX0+"
    "Pa+9viGqjT5fmRU/cJY9vwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFkw3v3yvyz4r"
    "Yug9dppvP7Mo8L6c3K4+AAAAAOhSs7/ZPKe+AAAAAH3xRL4NCW4/AAAAAAAAAACbtAM/AAAAAAAA"
    "AAC2H3s+AAAAAAAAAAAAAAAAGgw5vgAAAABZRPQ+V602PwAAAAAAAAAAAAAAAAAAAAAF7is/WB6C"
    "vgAAAAAAAAAA6Z6ovrtx+b4ZVP4+8NIavwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAO1Ww758IVs/"
    "AAAAAC38gb/P+0k/y8siv6eY9T6Y8dG+AAAAAIuMMD8AAAAAkOaFPgAAAAB1+Ti/AjaZP6+/pr4A"
    "AAAAAAAAAAAAAAAAAAAA7/EivwAAAAA4Fpe/+Gr/Pi0lXr/7vAI/98G4PgAAAAAkpK++AAAAAAAA"
    "AAAAAAAAAAAAAAAAAABLHj+/Jc4NvwAAAADOppq/AAAAAAAAAAAHS1I/WhgKAVgSEwoRCAESDQoH"
    "EgViYXRjaAoCCCBiGAoBWRITChEIARINCgcSBWJhdGNoCgIICEIECgAQFQ=="
)

_GOLDEN_ATTENTION_MERGED_QKV = (
    "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"
    "Lm1pY3Jvc29mdBAB"
)

_GOLDEN_MULTI_LAYERS_SHARED_INPUT = (
    "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"
    "BWJhdGNoCgIIFEIECgAQFQ=="
)

_GOLDEN_ONLY_OBSERVED_ACTIVATIONS = (
    "CAo6zwQKFgoCWDEKAlcxEgJZMSIGTWF0TXVsOgAKFgoCWDIKAlcyEgJZMiIGTWF0TXVsOgASAWcq"
    "jQIIEAgEEAFCAlcxSoACPOUAvxoZTT/2+nU/AAAAAAAAAAAAAAAAAAAAAAYiC7/fe34/AAAAAI54"
    "/z4AAAAAmf0BPwAAAAAAAAAAnbZFPwAAAAAAAAAAOSrsPsymEL9rCAa/jwRNPwAAAAAAAAAAAAAA"
    "APQcFL/m4xg/AAAAAHP75T5FTgs/AAAAAHwiHD8AAAAAAAAAAOZPPb8AAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAACcOF4/gTBKv3znoL8AAAAAzfDmPgAAAAChzJe/ZdTwvsIUK7/r"
    "XJK/AAAAAAAAAADFs+K+AAAAAAAAAACsmFY/3NboPn/WeT8AAAAAWm39PiqdAQgMCAMQAUICVzJK"
    "kAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABaGQoCWDESEwoRCAESDQoHEgViYXRjaAoC"
    "CBBaGQoCWDISEwoRCAESDQoHEgViYXRjaAoCCAxiGQoCWTESEwoRCAESDQoHEgViYXRjaAoCCARi"
    "GQoCWTISEwoRCAESDQoHEgViYXRjaAoCCANCBAoAEBU="
)

_GOLDEN_FP16_MATMUL = (
    "CAo62wEKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAEIEAgEEApCAVdKgAEAADq4QDiEuG+6AAC7"
    "Npi3AAAIPhK4AACxOMU2cjq/NgAAAAAAAAAAAAAWuAAAvLYAAAAAAABkvQAApLigOK25AADXtgAA"
    "KzkAAAAAAAAAAAAAFLEAACQ5MDn1ugAAAAAAAAAAAACeOOY4AAAhNwAAAAAAAKK5MLUAAPY5AACf"
    "PFoYCgFYEhMKEQgKEg0KBxIFYmF0Y2gKAggQYhgKAVkSEwoRCAoSDQoHEgViYXRjaAoCCARCBAoA"
    "EBU="
)

_GOLDEN_FP16_ATTENTION_MERGED_QKV = (
    "CAo63gcKRwoBWAoEV3FrdgoEQnFrdhIBWRIHcHJlc2VudCIJQXR0ZW50aW9uKhAKCW51bV9oZWFk"
    "cxgCoAECOg1jb20ubWljcm9zb2Z0EgFnKo8GCBAIGBAKQgRXcWt2SoAGAAAUtwAAAAAAAAAAAADx"
    "uAAAAAAAAAAAeTYAAAAAAAAAAAAAAAAAAGg7AAAAAAAAEbkAAAAAAAAAAAAAAAAAAAAAAAAAABQ1"
    "AAAAABU3AACWOAAAAAAAAAAAAAAAAA63AAAAAAAA97cAAAAAAAAAAFQ1xTeRNFU12TTftD25b7UA"
    "ACQ1JbZKNMs46q8AAAAAAAAAAAAAAAC7uAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWjgAAAAA"
    "AAC+NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIo0AAAAAAAAVTgAAAAAAAAAAAAAEra9tAAAAAAA"
    "AAAAAADPtK2oAAAAAAAAAAAAAAAAArZ9MBq3AAAAAAAAbzgAAJw0WyUQN/I4AAAAAAAAAAAAAAk3"
    "J7gAAAAAAAAAAAAA4bXutwAAAAA+t5C1AAAAAAAAAAAAACCyVzSrOYSzNLMnNJQ0AAAAAAQyo7MA"
    "AMqy1zVstgAAAACwt2W0AABuNRg4cDY9OHk2AACBsIe2iTSft0u4HzMjsHushrEAAF61QzQAAP82"
    "AADauAS0xjQAAAAAAABrNsmtAAAWOlG5AABusgAAAAAAAAAAYbEAAOA3rTZUNgAAMzNUuQAAAADa"
    "uH00e7T/OrU0AAAAAAAAS6/MN0CrprgZtAAAfad5tAAAAAA2rAAAkTKZMYM1ELEqMAAAs7QAACs4"
    "AACQtOa1AAClOwAAkDgAAN0pzbU+swAA1LZmODM0zreusXW0AAAAAAa3AADduL61AAB8MoG0AADt"
    "swAAuLsguPatAAAAAAAAAADeuQAAAABkuQAAwjgAABuoNzTUuQAAiaZyJgAA/LWWNY2sAAAAAHa5"
    "1DY9N0C1RrcAANs1AABptBgzU7f3NxW4QjK+t0q2VSzyNAAAAABPtpM3AADVslWwwq8AAN03pDYA"
    "AJuxSjiQuSyyhSD+qwAA+jQAACM0kbQAACE4AAAAACApmzQAADwxbTMAAEQ5bqVKORe1DzYAAGM2"
    "RTHStV84AAAAAPI5KjwIGBAKQgRCcWt2SjCzI7qi0amPLqGoJSAQpiMpMyOqpRKu8SA7qE8u7yhQ"
    "KBEpQCTiJhYk2JzMKVYpGiNaHwoBWBIaChgIChIUCgcSBWJhdGNoCgUSA3NlcQoCCBBiHwoBWRIa"
    "ChgIChIUCgcSBWJhdGNoCgUSA3NlcQoCCAhCBAoAEBVCEQoNY29tLm1pY3Jvc29mdBAB"
)


# --- Core: matches the pure-Python reference exactly ----------------------


def test_sparsegpt_pruning_cpp_matmul_unstructured_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=50)
    rng = np.random.default_rng(150)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    expected = _golden(_GOLDEN_MATMUL_UNSTRUCTURED)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    # Real recomputation happened, not a same-shape no-op.
    assert not np.array_equal(_weight(actual), _weight(model))


def test_sparsegpt_pruning_cpp_matmul_reaches_roughly_the_target_sparsity():
    K, N = 64, 16
    model, _w = _matmul_model(K=K, N=N, seed=58)
    rng = np.random.default_rng(158)
    x_cal = rng.standard_normal((96, K)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.1)
    # Value-only rewrite -- shape is never touched.
    assert _weight(pruned).shape == _weight(model).shape


def test_sparsegpt_pruning_cpp_gemm_transb_matches_python_reference():
    # transB=1 Gemm -- exercises the weight_transposed=True branch of both
    # the Python reference and this C++ port's own w <-> w_nk reshape.
    K, N = 24, 6
    rng = np.random.default_rng(51)
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.4  # [N, K], transB layout
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(bias, "B")],
    )
    x_cal = rng.standard_normal((40, K)).astype(np.float32)

    expected = _golden(_GOLDEN_GEMM_TRANSB)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4, proc_block_size=10
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    assert not np.array_equal(_weight(actual), w)


def test_sparsegpt_pruning_cpp_nm_pattern_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=52)
    rng = np.random.default_rng(152)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    expected = _golden(_GOLDEN_NM_PATTERN)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))

    # Exactly 2 of every 4 consecutive columns survive, per output row --
    # the actual N:M structural guarantee, not merely "matches Python".
    w_nk = _weight(actual).T  # [N, K]
    for row in w_nk:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            if len(group) == 4:
                assert np.count_nonzero(group) == 2


def test_sparsegpt_pruning_cpp_attention_merged_qkv_matches_python_reference():
    hidden = 16
    nq = nk = nv = 8
    total_n = nq + nk + nv
    num_heads = 2
    rng = np.random.default_rng(53)
    w_qkv = rng.standard_normal((hidden, total_n)).astype(np.float32) * 0.3
    bias = rng.standard_normal((total_n,)).astype(np.float32) * 0.05
    model = _model(
        f"""
        g (float[batch,seq,{hidden}] X) => (float[batch,seq,{nv}] Y)
        {{
          Y, present = com.microsoft.Attention <num_heads={num_heads}>(X, Wqkv, Bqkv)
        }}
        """,
        initializer=[_f32(w_qkv, "Wqkv"), _f32(bias, "Bqkv")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    rng2 = np.random.default_rng(153)
    x_cal = rng2.standard_normal((3, 5, hidden)).astype(np.float32)

    expected = _golden(_GOLDEN_ATTENTION_MERGED_QKV)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(expected))
    # The merged weight was actually pruned, and the bias (never touched by
    # SparseGPT -- see this port's own scope) is untouched.
    assert not np.array_equal(_weight(actual), w_qkv)
    np.testing.assert_array_equal(_weight(actual, index=1), bias)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_sparsegpt_pruning_cpp_multiple_layers_sharing_one_input_matches_python_reference():
    # Mirrors the shape com.microsoft::GroupQueryAttention's own separate
    # Q/K/V projections take: three independent MatMul weights, all reading
    # the SAME upstream activation -- pruning.py's own docstring explains
    # why these need no special-casing at all (ordinary MatMul/Gemm nodes,
    # ranked and pruned exactly like any other layer). Each gets its own H
    # (built from the one shared probe) and is pruned completely
    # independently -- exactly what this test checks, entry for entry
    # against the Python reference for all three weights at once.
    hidden = 20
    rng = np.random.default_rng(54)
    wq = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    wk = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    wv = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[batch,{hidden}] X) => (float[batch,{hidden}] Q, float[batch,{hidden}] K, float[batch,{hidden}] V)
        {{
          Q = MatMul(X, Wq)
          K = MatMul(X, Wk)
          V = MatMul(X, Wv)
        }}
        """,
        initializer=[_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv")],
    )
    rng2 = np.random.default_rng(154)
    x_cal = rng2.standard_normal((40, hidden)).astype(np.float32)

    expected = _golden(_GOLDEN_MULTI_LAYERS_SHARED_INPUT)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    onnx.checker.check_model(actual)
    for i in range(3):
        _assert_bytewise_close(_weight(actual, i), _weight(expected, i))
        assert not np.array_equal(
            _weight(actual, i), _weight(model, i)
        )  # each was actually pruned


# --- No-op / declined-input behavior ---------------------------------------


def test_sparsegpt_pruning_cpp_zero_sparsity_is_a_noop():
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=55)
    rng = np.random.default_rng(155)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_weight(pruned), w)


def test_sparsegpt_pruning_cpp_no_calibration_batches_leaves_layer_untouched():
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=56)
    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[], sparsity=0.5
    )
    np.testing.assert_array_equal(_weight(pruned), w)


def test_sparsegpt_pruning_cpp_only_layers_with_observed_activations_are_pruned():
    # Two independent MatMul layers, each fed by its OWN graph input --
    # calibration_data supplies real data for "X1" but only an all-zero
    # batch for "X2" (still required: ModelExecutor::Run needs every
    # positional graph input filled, so it can't simply be omitted). A
    # dead (all-zero) probe activation makes its Hessian's own diagonal
    # exactly zero everywhere, so every column of that layer is "dead"
    # (see InverseHessianCholesky's own dead-channel handling) -- but the
    # layer is still very much observed and processed (dead columns get a
    # fixed diagonal of 1.0 specifically so they CAN still be pruned/
    # compensated, not skipped), so both layers end up pruned. This checks
    # the two are pruned completely independently of one another: W1 (real
    # signal) reconstructs the layer well; W2 (dead input) is provably
    # equivalent to magnitude-only pruning on all-zero-Hessian columns.
    K1, N1 = 16, 4
    K2, N2 = 12, 3
    rng = np.random.default_rng(59)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((K2, N2)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K1}] X1, float[batch,{K2}] X2) => (float[batch,{N1}] Y1, float[batch,{N2}] Y2)
        {{
          Y1 = MatMul(X1, W1)
          Y2 = MatMul(X2, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    rng2 = np.random.default_rng(159)
    x1_cal = rng2.standard_normal((40, K1)).astype(np.float32)
    x2_cal = np.zeros((40, K2), dtype=np.float32)

    expected = _golden(_GOLDEN_ONLY_OBSERVED_ACTIVATIONS)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X1": x1_cal, "X2": x2_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual, 0), _weight(expected, 0))
    _assert_bytewise_close(_weight(actual, 1), _weight(expected, 1))
    assert not np.array_equal(_weight(actual, 0), w1)


def test_sparsegpt_pruning_cpp_requires_n_and_m_together():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], m=4)


def test_sparsegpt_pruning_cpp_sparsity_out_of_range_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], sparsity=1.5)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], sparsity=-0.1)


def test_sparsegpt_pruning_cpp_bad_nm_relationship_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        # n > m is never valid (n must be <= m).
        onnxsim.apply_sparsegpt_pruning_cpp(model, calibration_data=[], n=5, m=4)


# --- Conv (ordinary/depthwise/general-grouped): NEW this round -------------
#
# Unlike the MatMul/Gemm/Attention tests above, these do NOT compare against
# onnxsim.apply_sparsegpt_pruning (now an alias for this very port -- see
# this file's own module docstring) -- Conv is genuinely new coverage this
# round, so each test below instead builds its own independent reference:
# an independent nested-loop im2col patch unfold (never this port's own
# ConvIm2ColAccumulateHessian, nor pruning.py's own _conv_im2col_patches)
# feeding a from-scratch transliteration of the reference implementation's
# own `fasterprune` (_reference_sparsegpt, written fresh from
# https://github.com/IST-DASLab/sparsegpt) -- mirroring the exact
# verification bar tests/test_pruning.py's own SparseGPT-Conv section
# already established for the pure-Python implementation this ports.


def _reference_sparsegpt(w_nk, h, sparsity, n, m, percdamp, blocksize):
    # An independent transliteration of the reference implementation's own
    # SparseGPT.fasterprune (https://github.com/IST-DASLab/sparsegpt/blob/
    # master/sparsegpt.py) -- see tests/test_pruning.py's own
    # identically-named, independently-written function for the same
    # verification role there. Uses the reference's own prunen/prunem
    # naming (prunen = number pruned per group of prunem), the mirror image
    # of onnxsim's own n/m ("n kept per group of m") convention.
    w = w_nk.copy().astype(np.float64)
    rows, cols = w.shape
    h = h.copy().astype(np.float64)
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    damp = percdamp * np.mean(np.diag(h))
    diag = np.arange(cols)
    h[diag, diag] += damp
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T

    prunen = 0 if n is None else m - n
    prunem = 0 if m is None else m

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        w1 = w[:, i1:i2].copy()
        q1 = np.zeros_like(w1)
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        if prunen == 0:
            tmp = w1**2 / (np.diag(hinv1).reshape(1, -1)) ** 2
            thresh = np.sort(tmp.flatten())[int(tmp.size * sparsity)]
            mask1 = tmp <= thresh
        else:
            mask1 = np.zeros_like(w1, dtype=bool)

        for i in range(count):
            w_col = w1[:, i]
            d = hinv1[i, i]
            if prunen != 0 and i % prunem == 0:
                tmp = (
                    w1[:, i : i + prunem] ** 2
                    / (np.diag(hinv1)[i : i + prunem].reshape(1, -1)) ** 2
                )
                idx = np.argsort(tmp, axis=1)[:, :prunen]
                mask1[:, i : i + prunem] = False
                np.put_along_axis(mask1[:, i : i + prunem], idx, True, axis=1)
            q_col = w_col.copy()
            q_col[mask1[:, i]] = 0.0
            q1[:, i] = q_col
            err_col = (w_col - q_col) / d
            w1[:, i + 1 :] -= np.outer(err_col, hinv1[i, i + 1 :])
            err1[:, i] = err_col

        w[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return w


def _naive_conv_patches(
    x, kh, kw, pads, stride_h, stride_w, dilation_h=1, dilation_w=1
):
    # Brute-force, completely independent im2col unfold: an explicit
    # (in_channel, kh, kw)-ordered patch per output position, built with a
    # plain Python triple loop rather than any vectorized/strided-slice
    # unfolding -- mirrors tests/test_pruning.py's own
    # `_naive_conv_patch_hessian` oracle, adapted to return the patch
    # matrix itself (this file's own `_reference_sparsegpt` needs `H =
    # patches.T @ patches` fed in, not a pre-reduced Hessian).
    pad_top, pad_left, pad_bottom, pad_right = pads
    n, cin, in_h, in_w = x.shape
    xp = np.pad(
        x.astype(np.float64),
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )
    eff_kh = (kh - 1) * dilation_h + 1
    eff_kw = (kw - 1) * dilation_w + 1
    h_out = (xp.shape[2] - eff_kh) // stride_h + 1
    w_out = (xp.shape[3] - eff_kw) // stride_w + 1
    K = cin * kh * kw
    patches = np.zeros((n * h_out * w_out, K), dtype=np.float64)
    idx = 0
    for ni in range(n):
        for oh in range(h_out):
            for ow in range(w_out):
                col = 0
                for c in range(cin):
                    for i in range(kh):
                        for j in range(kw):
                            hh = oh * stride_h + i * dilation_h
                            ww = ow * stride_w + j * dilation_w
                            patches[idx, col] = xp[ni, c, hh, ww]
                            col += 1
                idx += 1
    return patches


def _reference_conv_sparsegpt(
    w,
    x,
    group,
    cin_per_group,
    kh,
    kw,
    pads,
    stride_h,
    stride_w,
    sparsity,
    n,
    m,
    percdamp=0.01,
    blocksize=128,
    dilation_h=1,
    dilation_w=1,
):
    # Builds the fully-independent expected pruned Conv weight for `group`
    # groups at once: each group's own channel-sliced patches
    # (_naive_conv_patches, never this port's own C++ machinery or
    # pruning.py's own _conv_im2col_patches) feed _reference_sparsegpt
    # (above) with that group's own correctly-sliced weight sub-block.
    cout = w.shape[0]
    filters_per_group = cout // group
    blocks = []
    for g in range(group):
        x_g = x[:, g * cin_per_group : (g + 1) * cin_per_group, :, :]
        patches = _naive_conv_patches(
            x_g, kh, kw, pads, stride_h, stride_w, dilation_h, dilation_w
        )
        h_g = patches.T @ patches
        w_nk_g = (
            w[g * filters_per_group : (g + 1) * filters_per_group]
            .astype(np.float64)
            .reshape(filters_per_group, cin_per_group * kh * kw)
        )
        expected_nk_g = _reference_sparsegpt(
            w_nk_g, h_g, sparsity, n, m, percdamp, blocksize
        )
        blocks.append(expected_nk_g.reshape(filters_per_group, cin_per_group, kh, kw))
    return np.concatenate(blocks, axis=0)


def test_sparsegpt_pruning_cpp_conv_ordinary_matches_independent_reference():
    Cin, Cout, kh, kw, spatial = 3, 6, 3, 3, 8
    rng = np.random.default_rng(200)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    out_sp = spatial - kh + 1
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}]>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(201)
    x_cal = rng2.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(actual)
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=1,
        cin_per_group=Cin,
        kh=kh,
        kw=kw,
        pads=(0, 0, 0, 0),
        stride_h=1,
        stride_w=1,
        sparsity=0.5,
        n=None,
        m=None,
        blocksize=12,
    )
    _assert_bytewise_close(_weight(actual), expected)
    assert not np.array_equal(_weight(actual), w)


def test_sparsegpt_pruning_cpp_conv_nm_pattern_matches_independent_reference():
    Cin, Cout, kh, kw, spatial = 4, 8, 3, 3, 8
    rng = np.random.default_rng(202)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    out_sp = spatial - kh + 1
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}]>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(203)
    x_cal = rng2.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(actual)
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=1,
        cin_per_group=Cin,
        kh=kh,
        kw=kw,
        pads=(0, 0, 0, 0),
        stride_h=1,
        stride_w=1,
        sparsity=0.0,
        n=2,
        m=4,
        blocksize=12,
    )
    _assert_bytewise_close(_weight(actual), expected)
    w_flat = _weight(actual).reshape(Cout, Cin * kh * kw)
    for row in w_flat:
        for start in range(0, len(row), 4):
            g = row[start : start + 4]
            if len(g) == 4:
                assert np.count_nonzero(g) == 2


def test_sparsegpt_pruning_cpp_conv_depthwise_matches_independent_reference():
    C, kh, kw, spatial = 6, 3, 3, 8
    group = C
    rng = np.random.default_rng(204)
    w = rng.standard_normal((C, 1, kh, kw)).astype(np.float32) * 0.5
    out_sp = spatial - kh + 1
    model = _model(
        f"""
        g (float[N,{C},{spatial},{spatial}] X) => (float[N,{C},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], group={group}>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(205)
    x_cal = rng2.standard_normal((3, C, spatial, spatial)).astype(np.float32)

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=6
    )
    onnx.checker.check_model(actual)
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=group,
        cin_per_group=1,
        kh=kh,
        kw=kw,
        pads=(0, 0, 0, 0),
        stride_h=1,
        stride_w=1,
        sparsity=0.5,
        n=None,
        m=None,
        blocksize=6,
    )
    _assert_bytewise_close(_weight(actual), expected)
    assert not np.array_equal(_weight(actual), w)


def test_sparsegpt_pruning_cpp_conv_grouped_matches_independent_reference():
    # General grouped (not depthwise, not group=1) Conv, with deliberately
    # different per-group calibration statistics (a bug sharing one Hessian
    # across groups, or mixing up which group's slice feeds which filter
    # rows, would silently pass on symmetric data but must fail here).
    group = 2
    Cin_per_group, Cout, kh, kw, spatial = 3, 8, 3, 3, 8
    Cin = Cin_per_group * group
    rng = np.random.default_rng(206)
    w = rng.standard_normal((Cout, Cin_per_group, kh, kw)).astype(np.float32) * 0.5
    out_sp = spatial - kh + 1
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], group={group}>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(207)
    x_cal = rng2.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)
    x_cal[:, :Cin_per_group, :, :] *= 8.0  # group 0 only: different scale

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=9
    )
    onnx.checker.check_model(actual)
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=group,
        cin_per_group=Cin_per_group,
        kh=kh,
        kw=kw,
        pads=(0, 0, 0, 0),
        stride_h=1,
        stride_w=1,
        sparsity=0.5,
        n=None,
        m=None,
        blocksize=9,
    )
    _assert_bytewise_close(_weight(actual), expected)
    assert not np.array_equal(_weight(actual), w)


def test_sparsegpt_pruning_cpp_conv_auto_pad_matches_independent_reference():
    Cin, Cout, kh, kw, spatial = 3, 5, 3, 3, 8
    rng = np.random.default_rng(208)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{spatial},{spatial}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], auto_pad="SAME_UPPER">(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(209)
    x_cal = rng2.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    # SAME_UPPER, kh=kw=3, stride=1: pad_total=2 -> (1, 1) split evenly.
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=1,
        cin_per_group=Cin,
        kh=kh,
        kw=kw,
        pads=(1, 1, 1, 1),
        stride_h=1,
        stride_w=1,
        sparsity=0.5,
        n=None,
        m=None,
    )
    _assert_bytewise_close(_weight(actual), expected)


def test_sparsegpt_pruning_cpp_conv_dilated_matches_independent_reference():
    Cin, Cout, kh, kw, spatial = 3, 5, 3, 3, 10
    rng = np.random.default_rng(210)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    eff_k = (kh - 1) * 2 + 1
    out_sp = spatial - eff_k + 1
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], dilations=[2,2]>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(211)
    x_cal = rng2.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=1,
        cin_per_group=Cin,
        kh=kh,
        kw=kw,
        pads=(0, 0, 0, 0),
        stride_h=1,
        stride_w=1,
        sparsity=0.5,
        n=None,
        m=None,
        dilation_h=2,
        dilation_w=2,
    )
    _assert_bytewise_close(_weight(actual), expected)


def test_sparsegpt_pruning_cpp_conv_reconstructs_better_than_a_same_mask_style_baseline():
    # The Conv analogue of the MatMul reconstruction test further down --
    # mirrors tests/test_pruning.py's own identically-named pure-Python
    # test: given comparable calibration signal, SparseGPT's Hessian-
    # compensated Conv weight should reconstruct the layer's real
    # (onnxruntime) output at least as well as naively zeroing the
    # same-shaped lowest-magnitude entries with no compensation at all.
    Cin, Cout, spatial = 4, 8, 10
    rng = np.random.default_rng(212)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    out_sp = spatial - 2
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[3,3]>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(1212)
    x_cal = rng2.standard_normal((16, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    K = Cin * 3 * 3
    w64 = w.astype(np.float64).reshape(Cout, K)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64).reshape(Cout, Cin, 3, 3)
    naive_model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = Conv<kernel_shape=[3,3]>(X, W)
        }}
        """,
        initializer=[_f32(w_naive.astype(np.float32), "W")],
    )

    def _run(m, x):
        sess = ort.InferenceSession(m.SerializeToString())
        return sess.run(None, {"X": x})[0]

    y_float = _run(model, x_cal)
    y_sparsegpt = _run(pruned, x_cal)
    y_naive = _run(naive_model, x_cal)
    err_sparsegpt = np.sum(
        (y_float.astype(np.float64) - y_sparsegpt.astype(np.float64)) ** 2
    )
    err_naive = np.sum((y_float.astype(np.float64) - y_naive.astype(np.float64)) ** 2)
    assert err_sparsegpt <= err_naive


def test_sparsegpt_pruning_cpp_conv_malformed_kernel_shape_left_untouched():
    # A kernel_shape disagreeing with the weight's own shape is malformed --
    # ResolveConvSpatialAttrs declines it, so the node is never even added
    # as a candidate; confirm the weight is left byte-identical, not
    # crashed on and not silently mismatched against the weight's own real
    # kernel size.
    Cin, Cout = 3, 4
    rng = np.random.default_rng(214)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          Y = Conv<kernel_shape=[2,2]>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(215)
    x_cal = rng2.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    np.testing.assert_array_equal(_weight(pruned), w)


def test_sparsegpt_pruning_cpp_fused_conv_matches_independent_reference():
    # FusedConv (com.microsoft) is matched exactly like a plain Conv --
    # _match_conv_weight_only's own op_type check, mirrored here.
    Cin, Cout, kh, kw, spatial = 3, 4, 3, 3, 8
    rng = np.random.default_rng(216)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    out_sp = spatial - kh + 1
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_sp},{out_sp}] Y)
        {{
          Y = com.microsoft.FusedConv<kernel_shape=[{kh},{kw}], activation="Relu">(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    rng2 = np.random.default_rng(217)
    x_cal = rng2.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    expected = _reference_conv_sparsegpt(
        w,
        x_cal,
        group=1,
        cin_per_group=Cin,
        kh=kh,
        kw=kw,
        pads=(0, 0, 0, 0),
        stride_h=1,
        stride_w=1,
        sparsity=0.5,
        n=None,
        m=None,
    )
    _assert_bytewise_close(_weight(actual), expected)
    assert not np.array_equal(_weight(actual), w)


# --- FLOAT16/BFLOAT16 weight matching ---------------------------------------
#
# FLOAT16/BFLOAT16 weights (both for the plain MatMul/vanilla-Gemm candidate
# path and for com.microsoft::Attention's own merged QKV weight, via the
# SparseGPT-local MatchAttentionProducerAnyFloat matcher) are IN scope for
# this C++ port -- IsSupportedFloatDtype/ReadTensorAsF64/WriteF64TensorAs,
# the same widening this file's own module docstring's "Scope" paragraph
# above already documents (now joined by Conv's own FLOAT32/FLOAT16/
# BFLOAT16 support, verified separately in the Conv section above rather
# than repeated here in FP16 form).
#
# BFLOAT16 gets no analogous real-onnxruntime-execution test here: this
# environment's onnxruntime has no CPU kernel for ANY op on a BFLOAT16
# tensor at all (confirmed the same way test_pruning.py's own "BFLOAT16 has
# no onnxruntime CPU execution support" section comment documents for the
# pure-Python reference -- a plain BFLOAT16 MatMul model raises
# NOT_IMPLEMENTED the moment a session is created), and this test file's own
# module docstring requires a real onnxruntime-backed executor throughout
# (never a fake/mock one) -- so a BFLOAT16 calibration run would fail on
# this environment's onnxruntime limitation alone, before this port's own
# widened matching/reading/writing code ever runs. There is accordingly no
# environment in which this file could exercise BFLOAT16 end to end; FLOAT16
# (which onnxruntime CAN execute) is tested below instead.


def test_sparsegpt_pruning_cpp_fp16_matmul_matches_python_reference():
    K, N = 16, 4
    rng = np.random.default_rng(61)
    w = (rng.standard_normal((K, N)).astype(np.float32) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[onnx.numpy_helper.from_array(w, "W")],
    )
    rng2 = np.random.default_rng(161)
    x_cal = rng2.standard_normal((32, K)).astype(np.float16)

    expected = _golden(_GOLDEN_FP16_MATMUL)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    w_expected = _weight(expected)
    w_actual = _weight(actual)
    assert w_actual.dtype == np.float16
    assert w_expected.dtype == np.float16
    _assert_bytewise_close(w_actual, w_expected)
    # Real recomputation happened, not a same-shape no-op.
    assert not np.array_equal(w_actual, w)


def test_sparsegpt_pruning_cpp_fp16_attention_merged_qkv_matches_python_reference():
    # Exercises MatchAttentionProducerAnyFloat specifically -- the
    # SparseGPT-local, dtype-widened duplicate of the shared (still
    # FLOAT32-only) MatchAttentionProducer used elsewhere in this file.
    hidden = 16
    nq = nk = nv = 8
    total_n = nq + nk + nv
    num_heads = 2
    rng = np.random.default_rng(62)
    w_qkv = (rng.standard_normal((hidden, total_n)).astype(np.float32) * 0.3).astype(
        np.float16
    )
    bias = (rng.standard_normal((total_n,)).astype(np.float32) * 0.05).astype(
        np.float16
    )
    model = _model(
        f"""
        g (float16[batch,seq,{hidden}] X) => (float16[batch,seq,{nv}] Y)
        {{
          Y, present = com.microsoft.Attention <num_heads={num_heads}>(X, Wqkv, Bqkv)
        }}
        """,
        initializer=[
            onnx.numpy_helper.from_array(w_qkv, "Wqkv"),
            onnx.numpy_helper.from_array(bias, "Bqkv"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    rng2 = np.random.default_rng(162)
    x_cal = rng2.standard_normal((3, 5, hidden)).astype(np.float16)

    expected = _golden(_GOLDEN_FP16_ATTENTION_MERGED_QKV)
    actual = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    w_expected = _weight(expected)
    w_actual = _weight(actual)
    assert w_actual.dtype == np.float16
    _assert_bytewise_close(w_actual, w_expected)
    assert not np.array_equal(w_actual, w_qkv)
    # Bias (never touched by SparseGPT) is untouched, dtype included.
    np.testing.assert_array_equal(_weight(actual, index=1), bias)


# --- End-to-end reconstruction quality --------------------------------------


def test_sparsegpt_pruning_cpp_reconstructs_better_than_a_same_mask_style_baseline():
    # The actual point of the technique, checked end to end through the C++
    # port specifically (mirrors test_pruning.py's own identically-named
    # pure-Python test): given comparable calibration signal, SparseGPT's
    # Hessian-compensated result should reconstruct the layer's output at
    # least as well, on that same calibration data, as simply zeroing the
    # same-shaped lowest-magnitude entries with no compensation at all.
    K, N = 48, 12
    rng = np.random.default_rng(62)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model, _w = _matmul_model(K=K, N=N, seed=62)
    assert np.array_equal(_weight(model), w)
    x_cal = rng.standard_normal((512, K)).astype(np.float32)  # well-conditioned H

    pruned = onnxsim.apply_sparsegpt_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    w_sparsegpt = _weight(pruned).astype(np.float64)

    w64 = w.astype(np.float64)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64)

    x64 = x_cal.astype(np.float64)
    y_orig = x64 @ w64
    err_sparsegpt = np.sum((y_orig - x64 @ w_sparsegpt) ** 2)
    err_naive = np.sum((y_orig - x64 @ w_naive) ** 2)
    assert err_sparsegpt <= err_naive
