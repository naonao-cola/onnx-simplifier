"""Tests for ``onnxsim.apply_wanda_pruning_cpp`` -- the C++-backed port of
``onnxsim.apply_wanda_pruning`` (Wanda, Sun et al., 2023; see
``onnxsim/structured_pruning_entry.cpp``'s own "Wanda unstructured
(element-wise) pruning" section and ``ApplyWandaPruning``). Like
``test_sparsegpt_pruning_cpp.py``/``test_structured_wanda_pruning_cpp.py``,
this runs the model over real calibration data through a real
``onnxruntime``-backed :class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor.

Unlike ``apply_sparsegpt_pruning_cpp`` (which recomputes every *kept*
entry's own value too), Wanda is a one-shot static importance score --
exactly like ``prune_magnitude_cpp`` -- so a correct port zeros exactly the
same entries the pure-Python ``onnxsim.apply_wanda_pruning`` reference
zeros, and every surviving entry is byte-identical to the original (never
recomputed).

Scope: this port matches plain ``MatMul``/vanilla-``Gemm`` (not
``com.microsoft::FusedGemm``/``GemmFastGelu``), ``com.microsoft::
Attention``'s merged QKV weight, each with a constant 2-D (1-D merged bias,
for Attention) FLOAT32/FLOAT16/BFLOAT16 weight, and every 2-D ``Conv``
node's constant 4-D FLOAT32/FLOAT16/BFLOAT16 weight -- ordinary
(``group=1``), fully depthwise (``group == in_channels == out_channels``),
and general grouped (``1 < group < in_channels``) alike -- via a
from-scratch im2col per-``(in_channel, kh, kw)``-tap activation norm
(``ConvPatchSqSum``/``ConvWandaCalibrationStats``) and a grouped/depthwise
group-relative norm expansion (``ConvGroupRelativeNorm``), mirroring
pruning.py's own ``_conv_patch_sq_sum``/``_conv_group_relative_norm``
exactly.

TRUE parity with the pure-Python ``onnxsim.apply_wanda_pruning`` (all three
candidate families, all three Conv `group` shapes) is now fully verified,
including a prior gap unrelated to Conv: this port's own
``WandaCalibrationStats`` (shared with ``ApplyStructuredWandaPruning``/
``ApplyAttentionHeadWandaPruning``) used to compute a per-channel-axis
activation L2-norm for a MatMul/Gemm candidate's activation at ANY rank >= 1
(probe axis -1, the same "reduce over every leading axis" handling it gives
a rank-3 Attention activation), whereas pruning.py's own
``_wanda_unstructured_calibration_stats`` explicitly requires ``x.ndim ==
2`` for its own (non-Attention, non-Conv) activation statistic and falls
back to plain magnitude importance for anything else (e.g. a rank-3
activation feeding a plain 2-D MatMul weight, which is exactly the shape a
batched/sequence MatMul input takes in practice) -- caught by
``test_wanda_pruning_falls_back_to_magnitude_without_matching_activation``
in ``tests/test_pruning.py``, still present below as
``test_wanda_pruning_cpp_matmul_rank3_activation_falls_back_to_plain_
magnitude``. That gap is now closed: ``WandaCalibrationStats`` takes a
``require_rank2`` set, populated (only by ``ApplyWandaPruning``, only for
its plain MatMul/Gemm candidates -- never Attention's, never the other two
callers') with exactly the probe names that must need `ndim == 2` --
mirroring pruning.py's own `act_norm`/`attn_act_norm` split exactly (see
``structured_pruning_entry.h``'s own ``ApplyWandaPruning`` declaration
comment for the full writeup).

``onnxsim.apply_wanda_pruning`` (the pure-Python name) is now itself a thin
alias for :func:`onnxsim.apply_wanda_pruning_cpp` (full parity verified --
see pruning.py's own "Wanda unstructured (element-wise) pruning" section
comment), exactly like ``apply_transformer_block_pruning``/
``apply_magnitude_pruning`` before it. Every test below that used to call
BOTH entry points and compare their live outputs would be tautological
(literally the same code path twice) if left as-is -- those now instead
compare the C++ port's output against a golden fixture captured from the
real pure-Python implementation *before* its own mutating body was replaced
by the alias -- see ``_GOLDEN_*`` below (base64-encoded serialized
``ModelProto`` bytes, inlined directly, mirroring
``test_transformer_block_pruning_cpp.py``'s own established convention --
see that file's own module docstring for why inline base64 rather than a
checked-in ``.onnx`` fixture file). The FLOAT16/BFLOAT16 tests confirm
those dtypes are matched, pruned, and written back with their own exact
original bit pattern preserved for every surviving entry -- not merely "not
crashed on".
"""

import base64

import ml_dtypes
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


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


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


def _magnitude_weight(model, node_index=0, input_index=1):
    # `onnxsim.apply_magnitude_pruning` is now a thin alias for the C++-backed
    # `onnxsim.prune_magnitude_cpp`, which leaves the original initializer
    # dangling and appends a new, anonymously-named one for the pruned
    # weight (see `tests/test_pruning_cpp.py`'s own `_weight` helper) --
    # unlike `_weight` above (position-based, still correct for
    # `apply_wanda_pruning_cpp`'s own in-place-mutating pure-Python
    # reference), a magnitude-pruning result must be resolved via the
    # node's CURRENT weight input.
    node = model.graph.node[node_index]
    w_name = node.input[input_index]
    init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(init)


def _assert_bytewise_close(actual, expected, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(
        actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
    )


# Frozen from onnxsim.apply_wanda_pruning's own real pure-Python
# implementation, on the exact model + calibration seed each corresponding
# test below builds, before that implementation's own mutating body was
# replaced by the thin C++ alias (see this file's own module docstring).
_GOLDEN_MATMUL_UNSTRUCTURED = (
    "CAo62wgKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAgIIAgIEAFCAVdKgAgAAAAAAAAAANdZqz7e"
    "kjG/1TtVPwAAAAAAAAAAAAAAAAAAAAAAAAAADLkrvwAAAAAAAAAAAAAAAKzJrL8vU0e/AAAAAAAA"
    "AAAAAAAA7P0Mv/LDl74AAAAAgLHDvwAAAACPA8O+AAAAAAAAAACnwxC/AAAAAAAAAAAAAAAAAAAA"
    "AClsBj8VSYA/AAAAAF/oz74AAAAA9uuFPwAAAAAAAAAA3cyjvhN+Cb99Lgy/wzi3P9Wptb5a3Bi/"
    "AAAAALNIvT59Qwu/AAAAAAAAAAAAAAAAPd0evwAAAAA9bmc/AAAAAJD6Fr8AAAAAK0W5PgAAAAAA"
    "AAAAAAAAAAAAAAAAAAAALHDPPioTZ78AAAAAAAAAAAAAAAAAAAAAsJ5XPwAAAAAAAAAAAAAAABFK"
    "2D4AAAAAbqUPv0tB3r4AAAAAo2dkPwVbwL64wNk+/msov18qzj4AAAAAAAAAAFJVNT+mPY0/AAAA"
    "AFRe9L4KrNa+AAAAAAAAAAAAAAAAD+TgvgAAAAAAAAAAi4cUPwAAAABi2EU/hN8BP5sH5T4AAAAA"
    "pGIdv46e6j6yOiw/AAAAAJ+FJ782Ng+/X6j5vlziDr/CqFw/AAAAAGn6Dj8H/La+k4X2PkTLoT6N"
    "CAU/AAAAAAAAAABgDt6+e9JFv4N3GL8AAAAAAAAAAK+l7r4AAAAAbUToPgAAAAAAAAAAAAAAAAAA"
    "AABBewI/AAAAAAAAAAAAAAAAjFW9PgAAAAAAAAAAdKAJvwAAAAAAAAAAOl47P3vfBL8AAAAASeFc"
    "PwYfiz+Rdpa/AAAAAAAFRr8AAAAAAAAAAAAAAACuqZQ/AAAAAAAAAAAebK0+6B75PgAAAAAiaRE/"
    "d5ohvwAAAAAAAAAAAAAAAAAAAAAAAAAAAACAP36OFb8AAAAAAAAAAA6tor+9SSy/AAAAAAAAAABg"
    "gTG/AAAAAAAAAAAAAAAAAAAAAAAAAADv8j4/kJEfv24rIj+Kww4/AAAAAOZwNj9FG5e+eVb5PuJS"
    "9j71vTK/AAAAANdrJb8AAAAARuTxvqiklb4eNNa+gNfPPppXqj4TAAw/jMTaPufq4L4AAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAABWLvg+VyPqvnRgNT8AAAAAAAAAAAAAAAAAAAAAAAAAAPqJQD8AAAAA"
    "AAAAAAu/1z4AAAAAvTk9vxkAVz8AAAAAhbgPPwAAAABsHac/AAAAAFIGxT6VMVc/XiyePgmIQL8A"
    "AAAAI0DLvgAAAAAAAAAA8U8MvwAAAACGsoQ/ka6SPwAAAADPpuI+AAAAAMaZEb8AAAAA0QXFvqzl"
    "pD437Qs/AAAAAGjgyD78OyM/fuwQvwAAAAAAAAAAAAAAAG0UXT8AAAAAWhgKAVgSEwoRCAESDQoH"
    "EgViYXRjaAoCCCBiGAoBWRITChEIARINCgcSBWJhdGNoCgIICEIECgAQFQ=="
)

_GOLDEN_GEMM_TRANSB = (
    "CAo6zgUKIwoBWAoBVwoBQhIBWSIER2VtbSoNCgZ0cmFuc0IYAaABAjoAEgFnKswECAYIGBABQgFX"
    "SsAEAAAAANjPfr4AAAAA0EAAPwAAAADdneG+uZSjvtECNj8AAAAAC2jyvgAAAAAAAAAAAAAAADvG"
    "Pj8AAAAALUYev9yBGj8AAAAAAAAAAMEQi74LGdS+7d4hPwZqG7/jAH6+3+DqvpfVEz8AAAAA/g2R"
    "vrMPkj4FHkc/ESvFvp/+tr4AAAAAAAAAAC+h6L4AAAAAAAAAAHX4sD6KSRi/AAAAAAAAAADP1FI/"
    "r3l1vwAAAAAAAAAAvXUdP2rO8b4AAAAAAAAAAAAAAAAAAAAA2O4EvwxTTD8Pgo6+AAAAAEeLyz4A"
    "AAAAkHlmPoabHb8AAAAAAAAAACbxIb+ebpG+RcvavmmCyD5hN7C+pwUmvwAAAAAAAAAAZ4mJP6f9"
    "Pj8AAAAAL7aqvrh4dT8AAAAAEo85vwAAAABHwTq/zm0FPwAAAAC9bP6+R1aKvsoLcr+bHjk/AAAA"
    "AC9giz7dxws/AAAAAAEZhb4AAAAAAAAAAN5iwb4AAAAAAAAAAAAAAADvZIa+Qhj6vgAAAAAAAAAA"
    "KM8PPzgj9z4AAAAA7UeJPgAAAACruCM/xXeyvpU3Aj8AAAAAAAAAAJxbiT9rDkw/oa0svwAAAAAA"
    "AAAAYFD5PpAsBr8AAAAAAAAAAEkqrL4bmbE+GjJqvgAAAADd8tm+X5pNPrWqXT6pbA6+nEq/vgAA"
    "AAAAAAAAAAAAAAAAAADgu+E+PqoavgAAAABTfS4/N9+9vgAAAAAAAAAAFwuqvtPiGr9ar9q+HdRv"
    "vwAAAAAAAAAAKiEIBhABQgFCShjT05s924xSPkqkljzGRq888O9ZPaOnX71aGAoBWBITChEIARIN"
    "CgcSBWJhdGNoCgIIGGIYCgFZEhMKEQgBEg0KBxIFYmF0Y2gKAggGQgQKABAV"
)

_GOLDEN_NM_PATTERN = (
    "CAo62wgKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAgIIAgIEAFCAVdKgAgAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAM2TEb+O+5M+kqkmvsFu/T5pFig/+vlQvyc2Kj+3VK6+2c8EvwAA"
    "AABhLhk++yMBP0+itD/+UDu/AAAAAIeeQD4AAAAAtRRcvwAAAAAAAAAAAAAAAAAAAABMzyw/AAAA"
    "ACleyz5dzgO/qkqjv43Txz4AAAAAnxqzPgAAAAAAAAAAAAAAAPygAz8AAAAAAAAAAAAAAAAAAAAA"
    "w+ONPxjhgr4veaI/AAAAAAAAAAAAAAAAS1jvPq6VWT9tzN6+AAAAAAAAAAAAAAAA5jf4vsuEaz4W"
    "lMc+AAAAAAAAAACAHwc+e6MBP8HEfD7eEQQ/AAAAAJaz3D44FrY+AAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAqYIJvwAAAAAElJC/QET7vsEbuj4AAAAAAAAAACgo7b7djj+/uOQH"
    "vwAAAAAAAAAApVtgPyhmcT88i90+JB/QvgAAAAC7aWa+AAAAAAAAAAAAAAAAAAAAAAAAAACzKmq/"
    "AAAAAObeqj4AAAAAXkgmvwCD/D0AAAAAxPQgvwAAAAAAAAAAAAAAAAc1Wz8AAAAA9orcPbUbZj4A"
    "AAAARrx4vkqiej4AAAAAXE2TP9bAJj8AAAAA/nIsP6hx8r4AAAAAes9MPwYKbr8igAY/AAAAACeD"
    "oL6+Gtc+lmgZv10a6j4AAAAAAAAAAJMZFb8AAAAAsuDQPgAAAADEE00/AAAAAFAciD8MKoA/AAAA"
    "AF7MVD8AAAAAAAAAAAAAAAASyl2/AAAAAAAAAAAAAAAAutiMvwAAAABlTwE/AAAAAAAAAACqXJA+"
    "AAAAAAAAAACS1Ng+WApEvwAAAAAAAAAAAAAAAAAAAAB7UJC+QMSyPgAAAAAAAAAAeItQv8UpPD4q"
    "YX8+79p9P1ztr74AAAAAAAAAALcpqr8AAAAA5q1cPq+Fqr6wC1Q/AAAAADdtKr5BYAc/AAAAAI8K"
    "BT8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT6VjPwAAAAAAAAAA5466PgAAAAA+VkI/AAAA"
    "APRgyr4AAAAA9vIIvwAAAAC9UMI+k5hkvwAAAABxmHO+AAAAAAAAAAAAAAAA/kmcPgAAAADlLTA/"
    "aLBlPrTNcr/6pCQ/3EEhv2Q5GD9qwvK+AAAAAPkbDT8AAAAAsM43PgJZnz7K1mO/HShuPwAAAAAA"
    "AAAAAAAAADRY6j4AAAAAsrsbvwAAAAAwt5G/xmDmPvKFXb+2dw4/AAAAAAAAAAAAAAAAAAAAAAAA"
    "AADdYKU+AAAAAAAAAABg3uS+ZT+gvgAAAABMC36/AAAAAAAAAACx6CA/WhgKAVgSEwoRCAESDQoH"
    "EgViYXRjaAoCCCBiGAoBWRITChEIARINCgcSBWJhdGNoCgIICEIECgAQFQ=="
)

_GOLDEN_ATTENTION_MERGED_QKV = (
    "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"
    "Lm1pY3Jvc29mdBAB"
)

_GOLDEN_MULTIPLE_LAYERS_SHARING_INPUT = (
    "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"
    "BWJhdGNoCgIIFEIECgAQFQ=="
)

_GOLDEN_GLOBAL_SPARSITY = (
    "CAo6vwUKFgoCWDEKAlcxEgJZMSIGTWF0TXVsOgAKFgoCWDIKAlcyEgJZMiIGTWF0TXVsOgASAWcq"
    "zQIIFAgEEAFCAlcxSsACAAAAAAAAAAAAAAAAAAAAAGm/Mz8Jczu/AAAAAAAAAAAAAAAAAAAAAAAA"
    "AAB2MUI/vvMOvwAAAADuSgU/AAAAAJfGQj8AAAAAAAAAAAAAAAAAAAAA2N8OPwAAAAAAAAAAdlk/"
    "PwAAAAAAAAAAAAAAAIrjTr8AAAAAAAAAAAAAAABZalm/AAAAAAAAAADagSA/AAAAAAAAAAAAAAAA"
    "AAAAAAAAAADIoSs/AAAAAAAAAACX0C6/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHuFQj+X"
    "mgy/AAAAAAAAAAD4uSm/AAAAAAAAAABmiY4/AAAAAAAAAADqxwO/r54aPwAAAAA1eiS/AAAAAAAA"
    "AAAAAAAAM9YjvzVoKD8AAAAA0PNJvzrv3L4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAqzQEICAgG"
    "EAFCAlcySsABDZwSwFwtXr8AAAAA4cQJwNZlaEAAAAAAnjsjQOrK6T+tI5tAaIKWv5lnRj8AAAAA"
    "KEtIv/puEUDpOw9AzgU/vwXpnD8v2iTAENjoP3gEvr9CC6q/JuZSQJ+rab+kyg8/PMENQAAAAAAA"
    "AAAAaaW5vwAAAACVc9M/AAAAAAoUJ8DHzt6/iWQfQBRgNcDjLXo/6vv7Pqbg0D/DceA/vptBP76j"
    "QL+2S1jAL3fBP8LItb+OVCi/zYRBPwAAAAA8DsK/WhkKAlgxEhMKEQgBEg0KBxIFYmF0Y2gKAggU"
    "WhkKAlgyEhMKEQgBEg0KBxIFYmF0Y2gKAggIYhkKAlkxEhMKEQgBEg0KBxIFYmF0Y2gKAggEYhkK"
    "AlkyEhMKEQgBEg0KBxIFYmF0Y2gKAggGQgQKABAV"
)

_GOLDEN_CONV_ORDINARY_GROUP1 = (
    "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"
)

_GOLDEN_CONV_FULLY_DEPTHWISE = (
    "CAo61wMKWwoBWAoBVxIBWSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUAB"
    "QAFAAaABByoQCgdzdHJpZGVzQAFAAaABByoMCgVncm91cBgIoAECOgASAWcqsAIICAgBCAMIAxAB"
    "QgFXSqACdYZFPwAAAABaJFS/+wukvtPVMD8AAAAAAAAAAAAAAAAAAAAAbSXNPgAAAAD7uQY/AAAA"
    "AAAAAAAAAAAAAAAAACI1Uz9zJCy/XDAevwAAAACVsec+AAAAAAAAAAAAAAAA/rsRvyCX+b4AAAAA"
    "AAAAAAAAAAAAAAAAioQFPwAAAAAYBA4/AAAAAAMFZz/V8xW/EChcvwAAAAAAAAAAAAAAANS/sD7G"
    "cRu/AAAAAPgVhD8AAAAAHx8zPwAAAAAAAAAAB2fhvgAAAAACbiW/I6AnPwAAAAAAAAAA3OfDPpd+"
    "zL4AAAAAAAAAAIdl4T4AAAAAZ4AXPwAAAAAAAAAAkxEZPwAAAAAAAAAA/ax7vwAAAAAAAAAAAAAA"
    "AA3r6T5jHWY/Wh8KAVgSGgoYCAESFAoEEgJOYgoCCAgKAxIBSAoDEgFXYiEKAVkSHAoaCAESFgoE"
    "EgJOYgoCCAgKBBICSDIKBBICVzJCBAoAEBU="
)

_GOLDEN_CONV_GENERAL_GROUPED = (
    "CAo6lwgKWwoBWAoBVxIBWSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUAB"
    "QAFAAaABByoQCgdzdHJpZGVzQAFAAaABByoMCgVncm91cBgEoAECOgASAWcq8AYIDAgCCAMIAxAB"
    "QgFXSuAGKoU5PwAAAAAYVt6+AAAAAAoN7L40cRo/HzS0PgAAAADuXhA/AAAAAB6RBr8AAAAAAAAA"
    "AAI2r74AAAAAkL0WPwAAAAAAAAAAAAAAAAAAAACqfdE+AAAAAD1TDz//8KW+AAAAAMHgpz4AAAAA"
    "AAAAAAAAAAAAAAAAAAAAAOf0cj5uFg4/d7s2v7spar9nNJo+fxajvgwhFT8P6pI/AAAAACogHb4A"
    "AAAAAAAAAAAAAADTGLm+AAAAACLCTr8AAAAAAAAAAMT2BT8AAAAAVs5GPvLPwb4AAAAA0CCfPtAa"
    "0T4AAAAAKVgiP+ybAj8AAAAA8rB8v83H4L4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABw11m/AAAA"
    "AFV/0D5WQa6+zgCDPgAAAAAAAAAAAAAAAMTou74AAAAAE6AZPwAAAAAnwuG+1+MVP+BoC79hLSk/"
    "AAAAAP8ofT4AAAAAAAAAAMjc3D4AAAAAYmenvgAAAACrqzK/AAAAAAZrNb8AAAAA2mSVPgAAAAAA"
    "AAAA+uqJvwAAAAA0iAM/AAAAAAAAAADIzYS+kosjP2quVb8AAAAAAAAAAAAAAAAAAAAAAAAAAEj7"
    "aL93eUw/AAAAAKYtgL53sA4/UybjvgAAAAA1Ac8+e7agPgAAAAAAAAAAsTHLvgAAAADEqwG/f/nc"
    "PoGftD4AAAAAw+dBvwAAAAAAAAAAGjPYvgAAAAAAAAAAAAAAAAAAAAAicaO+AAAAAA/zwb5fiDc/"
    "NOI9P+jjTr8AAAAAFnEhP4vlED9xiEo/VBcCPwAAAAAAAAAAMhsQPwAAAADRbCa/AAAAAMkcNL+J"
    "r6s+AAAAAAAAAAAAAAAAlsAuvwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPkiDb+DAfq+5+bnPlf8"
    "q74AAAAAieGkPgAAAABIcF4/cfa6vgAAAACya06/lKSsPgAAAAAAAAAAAAAAAFP8eT8AAAAAAAAA"
    "AAAAAAAAAAAAx4pCvwAAAABXAh2/Y1XFvue+9T6IHLq+AAAAAMfTqz48hx2/AAAAAF9u9b4AAAAA"
    "AAAAAPP1zj4AAAAAAAAAAF7YEz8AAAAAuWkJP+WQkz7fC78+U8KRvgAAAAAAAAAAKMqaPj2F6T4A"
    "IdI+AAAAAAAAAAAAAAAAWh8KAVgSGgoYCAESFAoEEgJOYgoCCAgKAxIBSAoDEgFXYiEKAVkSHAoa"
    "CAESFgoEEgJOYgoCCAwKBBICSDIKBBICVzJCBAoAEBU="
)

_GOLDEN_CONV_GENERAL_GROUPED_NM = (
    "CAo6lwgKWwoBWAoBVxIBWSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUAB"
    "QAFAAaABByoQCgdzdHJpZGVzQAFAAaABByoMCgVncm91cBgEoAECOgASAWcq8AYIDAgCCAMIAxAB"
    "QgFXSuAGAAAAAAAAAACgD8E+9WInPxOPwL46smY/AAAAAAAAAAAUm7C+AAAAAAAAAABY0xq/g2ZR"
    "vwAAAAAAAAAAgLMfP8V/WT8AAAAA3zoRPwAAAAA3yks/AAAAAOfINb84QHK+AAAAAAAAAAAAAAAA"
    "BN9FvwAAAAAZU5c/AAAAAJNwmr4AAAAAvCEKvxljmz4AAAAAAAAAACJ26j67xk8/AAAAAAAAAAAI"
    "fkG/33HePgAAAAArRkU+AAAAAAAAAAD12dw+mnaBvQAAAABPv/C9AAAAADaypz4AAAAAOPJTvtDJ"
    "mb4AAAAAAAAAAAAAAAAAAAAAOHqnPi9ejb6yUnC+X4O+PgAAAAAAAAAAbKULvwAAAAAQmm0/AAAA"
    "AAAAAABbcvU92t7Vvtt3tr4AAAAAAAAAABhoYb+HiGg/AAAAAAAAAADlMXe/y2QBvwAAAAAAAAAA"
    "11INv6UAIL8AAAAAAAAAAAAAAADHaro+AAAAAAVsIT+V1XQ+AAAAAEecsL6g/9k+AAAAAAAAAAAq"
    "6Ac/AAAAAL7ajD4AAAAAAAAAANqh3b4AAAAAF6APP9YFAT4AAAAAAAAAAAAAAADrDCk/ap7VPrvf"
    "qD0AAAAAWPOfPgAAAAAnBck+AAAAAOot3j4AAAAAj+/XvqxSJD8AAAAAAAAAAAAAAAAojlU+AAAA"
    "AAAAAAAKhfe+zyf2vu5ZiL4yUH8+AAAAAAAAAADbE6A+AAAAAAAAAACN94e+AAAAAHDYWr4AAAAA"
    "GyCuvlW+K74AAAAAeDkgvrhyTj4AAAAAAAAAAAAAAAAAAAAARF8CP3AWHL8AAAAAPwoHP4oUID8A"
    "AAAAYLazPg4VwD4AAAAAAAAAAPMgwj4AAAAAAAAAAMD8/z0AAAAAphkRPxmAk74AAAAAQ8afPgAA"
    "AAATVa6+AAAAAAAAAABYb+A+E9fUvko8Rr4AAAAAAAAAAADNZT4AAAAAAAAAAG8S7b4AAAAAu3go"
    "P0cMDj8AAAAAtvDAvgAAAAAfx8A+794evwAAAAAAAAAAO/VBvwAAAAAAAAAACkJePwAAAAABE4E+"
    "AAAAAFxmKz8AAAAAWqBSPhVcT77DQyC/AAAAAAAAAABM84S+v8WzvgAAAAAAAAAAAAAAAEJRiD8A"
    "AAAAiBPwPgAAAADHfyQ/Wh8KAVgSGgoYCAESFAoEEgJOYgoCCAgKAxIBSAoDEgFXYiEKAVkSHAoa"
    "CAESFgoEEgJOYgoCCAwKBBICSDIKBBICVzJCBAoAEBU="
)

_GOLDEN_CONV_DEPTHWISE_FLOAT16 = (
    "CAo6iQIKSQoBWAoBVxIBWSIEQ29udioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUAB"
    "QAFAAaABByoMCgVncm91cBgGoAECOgASAWcqewgGCAEIAwgDEApCAVdKbAQ2AAAvOwQ3AAAAAAAA"
    "AAArugAAMbgAAAAAEzcAAJA58zUAAAAAAABZO8u1AADdtgAALDgAAKU1SDcAAFy1AAAAAAAAVboA"
    "AAAAAAAAAI+4SjV7NwAAAABUuDA5AAAAAJw6AAAIugAAAAAXOlodCgFYEhgKFggKEhIKBBICTmIK"
    "AggGCgIICgoCCApiHQoBWRIYChYIChISCgQSAk5iCgIIBgoCCAoKAggKQgQKABAV"
)

_GOLDEN_MATMUL_FLOAT16 = (
    "CAo62wEKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAEIEAgEEApCAVdKgAGJtDq4QDiEuB67AAC7"
    "Npi3hrSPPRK4AAC3OPc1cjoAAAAAAABntAAAAADsuAAAAAAAAAAAAAAxvQAAsbmwOLu4JbSutQAA"
    "PTgAAAAAAAAAAAAAAAAAAOM5eDjSugAAAAAAAAAAAAAQOFY4AADeNgAAnzIAAA26AAAAAMQ4AAA1"
    "PFoYCgFYEhMKEQgKEg0KBxIFYmF0Y2gKAggQYhgKAVkSEwoRCAoSDQoHEgViYXRjaAoCCARCBAoA"
    "EBU="
)

_GOLDEN_MATMUL_BFLOAT16 = (
    "CAo62wEKEwoBWAoBVxIBWSIGTWF0TXVsOgASAWcqjAEIEAgEEBBCAVdKgAEDvz2/AAAAAAAAAAAA"
    "AIS/Gb8AAAe/AAAtPwa/AAAAAAAAAAB4Psy+xj8AALg+AABFv5M+AADxPgAAAADKvgAAAAAAAAAA"
    "Ij8AAAAAQT8AAIk/AACGPgAAAADgPgAAXL8AAMe+AACGvwAAeT4AAAAAIj9DP90+Az8TPxe/jb8D"
    "v1oYCgFYEhMKEQgQEg0KBxIFYmF0Y2gKAggQYhgKAVkSEwoRCBASDQoHEgViYXRjaAoCCARCBAoA"
    "EBU="
)

_GOLDEN_ATTENTION_MERGED_QKV_FLOAT16 = (
    "CAo63gcKRwoBWAoEV3FrdgoEQnFrdhIBWRIHcHJlc2VudCIJQXR0ZW50aW9uKhAKCW51bV9oZWFk"
    "cxgCoAECOg1jb20ubWljcm9zb2Z0EgFnKo8GCBAIGBAKQgRXcWt2SoAG1jKTOfi3AADrNSE1wDYA"
    "AAAAyjmANGU2AACuuAAAAACZNQAAAAAAAA44AAAAAAAAAAAAAAAAPzg9NQAAKTQAAAAAAAAAAAAA"
    "AAAAAAAAt7cAAAAAozYAAAAAAAAAAAAAYzIAAAAAAAAAAJm2AAAAAAAAizRBNw4zAAAAACw42LIA"
    "APsxNLQxOxU2AACBswAAAAAGtQQ0qDZltgAAFDSXtwAAAAA2uwAATrcAAAAAMbYAAB43AAAAAAA3"
    "ajQAAAAAAACDtOs3VzRltQAAgrd4tTc5nzLjNiC2AABsuc6xAAAyt8UyAAANtY80AAD+Mwm0vLUA"
    "tgAAAAAAAAg0/7bbMwAAAAAAAAAAbrUAAHO5AAAAAAAAAAAAAAAA77XUNrgzAAAAAAAA3jQAAPmy"
    "AADmtPc1sTQAAAAARDVlOQAA9TQAAAAACTXftke0AACJNFC1vDMAAAAAcbYAAAAAAAAAAAq46LNl"
    "tAAA+TUAAAAAAAA1NlO4c7MAAAAAErLNtdSyAAAAAAAAAAAAAAAAAABktEC5+bUAAAAAhrcuOeS1"
    "OrUAAAAAAAAiNUm3AAAAAAAAAAAoNQAAAAC5MQAAAAAAAAAAAjTDOFq4sTYAANkyLjUAAAAAAAAA"
    "AAAAA7ZmtQAAg7i2tgAAAADqtls4xTQAAAAAAAAAAHW1J7oAAAAAAACaNNkwAAAAAAAA2DMAAAAA"
    "AAArtC60vTMAACcxBbbptWQ5PrgAABa2AAAAAPmyAACqsgAAjDaKM7E0f7gzNS02AADttMG73rgA"
    "AAAAAAAAAAAAAACfNgAAPznONwAAAAAAAAAAZzhdtAAAAAAAAPQ3lrMAAB01AAAAAHc5AAAAAEe4"
    "AABptgAAAABUNcoxAADzO6ewHrbJs9G1wzJGOwAAfLYAAPC3AADAtAAAibUeNKA4AAAAAEg3AAAL"
    "tQAAkLSRtAAxAAAAAAAAAAAAAHG1bLQAADg03jQUs94xAAAUMey4AAAAABWxAACXOF0x/7F9tt00"
    "9rcutwAAxDQAAKQ4KjwIGBAKQgRCcWt2SjD/Jo6mHaUkrOYpDakXLF8rtiUDr0sp6KQ+LJuYWCnB"
    "nbMiO6W5omQrEag0qQgtKydaHwoBWBIaChgIChIUCgcSBWJhdGNoCgUSA3NlcQoCCBBiHwoBWRIa"
    "ChgIChIUCgcSBWJhdGNoCgUSA3NlcQoCCAhCBAoAEBVCEQoNY29tLm1pY3Jvc29mdBAB"
)


# --- Core: matches the pure-Python reference exactly ----------------------


def test_wanda_pruning_cpp_matmul_unstructured_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=50)
    rng = np.random.default_rng(150)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    golden = _golden(_GOLDEN_MATMUL_UNSTRUCTURED)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(golden))
    assert not np.array_equal(_weight(actual), _weight(model))


def test_wanda_pruning_cpp_matmul_reaches_roughly_the_target_sparsity():
    K, N = 64, 16
    model, _w = _matmul_model(K=K, N=N, seed=58)
    rng = np.random.default_rng(158)
    x_cal = rng.standard_normal((96, K)).astype(np.float32)

    pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.1)
    assert _weight(pruned).shape == _weight(model).shape


def test_wanda_pruning_cpp_gemm_transb_matches_python_reference():
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

    golden = _golden(_GOLDEN_GEMM_TRANSB)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(golden))
    assert not np.array_equal(_weight(actual), w)


def test_wanda_pruning_cpp_nm_pattern_matches_python_reference():
    K, N = 32, 8
    model, _w = _matmul_model(K=K, N=N, seed=52)
    rng = np.random.default_rng(152)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    golden = _golden(_GOLDEN_NM_PATTERN)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], n=2, m=4
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(golden))

    # Exactly 2 of every 4 consecutive columns survive, per output row --
    # the actual N:M structural guarantee, not merely "matches Python".
    w_nk = _weight(actual).T  # [N, K]
    for row in w_nk:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            if len(group) == 4:
                assert np.count_nonzero(group) == 2


def test_wanda_pruning_cpp_attention_merged_qkv_matches_python_reference():
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

    golden = _golden(_GOLDEN_ATTENTION_MERGED_QKV)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual), _weight(golden))
    assert not np.array_equal(_weight(actual), w_qkv)
    np.testing.assert_array_equal(_weight(actual, index=1), bias)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_multiple_layers_sharing_one_input_matches_python_reference():
    # Mirrors the shape com.microsoft::GroupQueryAttention's own separate
    # Q/K/V projections take: three independent MatMul weights, all reading
    # the SAME upstream activation -- ranked and pruned completely
    # independently (each has its own weight, but they share one act_norm
    # entry, keyed by x_name).
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

    golden = _golden(_GOLDEN_MULTIPLE_LAYERS_SHARING_INPUT)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.4
    )
    onnx.checker.check_model(actual)
    for i in range(3):
        _assert_bytewise_close(_weight(actual, i), _weight(golden, i))
        assert not np.array_equal(_weight(actual, i), _weight(model, i))


# --- global_sparsity mode ---------------------------------------------------


def test_wanda_pruning_cpp_global_sparsity_matches_python_reference():
    K1, N1 = 20, 4
    K2, N2 = 8, 6
    rng = np.random.default_rng(70)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((K2, N2)).astype(np.float32) * 2.0
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
    rng2 = np.random.default_rng(170)
    x1_cal = rng2.standard_normal((40, K1)).astype(np.float32)
    x2_cal = rng2.standard_normal((40, K2)).astype(np.float32)

    golden = _golden(_GOLDEN_GLOBAL_SPARSITY)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model,
        calibration_data=[{"X1": x1_cal, "X2": x2_cal}],
        sparsity=0.5,
        global_sparsity=True,
    )
    onnx.checker.check_model(actual)
    _assert_bytewise_close(_weight(actual, 0), _weight(golden, 0))
    _assert_bytewise_close(_weight(actual, 1), _weight(golden, 1))
    # The whole-model pooled sparsity target is reached exactly at the
    # element count level (no per-row floor), matching the Python original.
    total = w1.size + w2.size
    zeros = np.count_nonzero(_weight(actual, 0) == 0) + np.count_nonzero(
        _weight(actual, 1) == 0
    )
    assert zeros == pytest.approx(total * 0.5, abs=1)


def test_wanda_pruning_cpp_global_sparsity_rejects_nm():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(
            model, calibration_data=[], n=2, m=4, global_sparsity=True
        )


# --- No-calibration-data behavior: falls back to plain magnitude -----------


def test_wanda_pruning_cpp_no_calibration_batches_falls_back_to_plain_magnitude():
    # pruning.py's own apply_wanda_pruning falls back to plain-|W| magnitude
    # importance when no calibration activation was ever observed for a
    # layer (_wanda_importance's own `norm is None` branch) -- VERIFIED by
    # reading the Python source, not assumed. With calibration_data=[], this
    # C++ port should therefore match onnxsim.apply_magnitude_pruning
    # exactly, not merely leave the layer untouched (unlike
    # apply_sparsegpt_pruning_cpp, which has no data-free fallback at all).
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=56)

    pruned = onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=0.5)
    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), _magnitude_weight(magnitude_pruned))
    # Actually pruned, not a no-op.
    assert not np.array_equal(_weight(pruned), w)


# --- Rank-2-only MatMul/Gemm activation gate --------------------------------
#
# pruning.py's own `_wanda_unstructured_calibration_stats` requires a plain
# (non-Attention, non-Conv) MatMul/Gemm candidate's own activation to be
# EXACTLY rank 2 (`if x.ndim != 2: continue`) before it counts as an
# observed calibration activation at all -- a rank-3 batched/sequence
# activation feeding a plain 2-D MatMul weight is exactly as "unobserved" as
# no calibration_data at all, and falls back to plain magnitude. This is
# NOT a universal rule: pruning.py's own separate `attn_act_norm` statistic
# for `com.microsoft::Attention`'s merged QKV weight accepts any rank >= 2
# (its own `X` is always rank-3 `[batch, seq, hidden]` by construction), and
# neither `apply_structured_wanda_pruning`'s nor
# `apply_attention_head_wanda_pruning`'s own calibration statistics have any
# rank restriction at all. `WandaCalibrationStats`' own `require_rank2` set
# (structured_pruning_entry.cpp) reproduces exactly this MatMul/Gemm-only
# restriction -- these two tests cross-check both sides of that distinction
# directly against `onnxsim.apply_magnitude_pruning` (a real oracle
# independent of the Wanda calibration machinery itself, now that
# `onnxsim.apply_wanda_pruning` is this same C++ port).


def test_wanda_pruning_cpp_matmul_rank3_activation_falls_back_to_plain_magnitude():
    # Mirrors tests/test_pruning.py's own
    # test_wanda_pruning_falls_back_to_magnitude_without_matching_activation
    # -- the regression that originally caught WandaCalibrationStats
    # observing a MatMul/Gemm candidate's activation at any rank, not just
    # rank 2. The graph input itself is declared rank 3 (`[batch, seq, K]`)
    # -- a real batched/sequence MatMul input, not merely an
    # oddly-shaped calibration array fed to a rank-2-declared graph.
    K, N = 32, 8
    rng = np.random.default_rng(90)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    rng2 = np.random.default_rng(190)
    x_cal = rng2.standard_normal((2, 4, K)).astype(np.float32)  # rank 3, not 2

    pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), _magnitude_weight(magnitude_pruned))
    # Actually pruned, not a no-op.
    assert not np.array_equal(_weight(pruned), w)


def test_wanda_pruning_cpp_attention_rank3_activation_is_not_gated_like_matmul():
    # Contrast case: `com.microsoft::Attention`'s own rank-3 activation must
    # NOT be gated by the same rank-2-only restriction the test above
    # exercises for plain MatMul/Gemm -- it should still get a REAL
    # activation-weighted importance (differing from plain magnitude), never
    # the plain-magnitude fallback, confirming `require_rank2` was built
    # from only the MatMul/Gemm candidates' own `x_name`s, never the
    # Attention candidate's.
    hidden = 8
    nq = nk = nv = 4
    total_n = nq + nk + nv
    num_heads = 1
    salient = (0, 3)
    rng = np.random.default_rng(91)
    # Every non-salient entry's |W| is bounded well away from zero (>= 0.2)
    # and every salient entry's |W| is a full order of magnitude smaller
    # (~0.01, jittered so no two entries exactly tie at the plain-magnitude
    # pruning threshold) -- constructed with an explicit magnitude floor on
    # each side, rather than relying on a random draw happening to keep them
    # separated, so plain-magnitude pruning is GUARANTEED to drop every
    # salient entry first regardless of random seed.
    signs = rng.choice([-1.0, 1.0], size=(hidden, total_n))
    w_qkv = (signs * (0.2 + rng.random((hidden, total_n)) * 0.1)).astype(np.float32)
    w_qkv[salient, :] = (
        signs[salient, :] * (0.01 + rng.random((len(salient), total_n)) * 0.001)
    ).astype(np.float32)
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
    x_cal = rng.standard_normal((2, 5, hidden)).astype(np.float32)  # rank 3
    # ...but scale those same input channels' own activation up massively,
    # so Wanda should keep at least one of their entries despite the small
    # |W|, exactly like test_wanda_pruning_cpp_activation_weighting_differs_
    # from_plain_magnitude's plain-MatMul version of this same check.
    for c in salient:
        x_cal[:, :, c] *= 1000.0

    wanda_pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)

    w_wanda = _weight(wanda_pruned)
    w_magnitude = _magnitude_weight(magnitude_pruned)
    assert not np.array_equal(w_wanda, w_magnitude)
    for c in salient:
        assert np.count_nonzero(w_magnitude[c, :]) == 0
        assert np.count_nonzero(w_wanda[c, :]) > 0


def test_wanda_pruning_cpp_zero_sparsity_is_a_noop():
    K, N = 16, 4
    model, w = _matmul_model(K=K, N=N, seed=55)
    rng = np.random.default_rng(155)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_weight(pruned), w)


def test_wanda_pruning_cpp_requires_n_and_m_together():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], m=4)


def test_wanda_pruning_cpp_sparsity_out_of_range_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=1.5)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=-0.1)


def test_wanda_pruning_cpp_bad_nm_relationship_raises():
    model, _w = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], n=5, m=4)


# --- Activation-weighted importance provably differs from plain magnitude --


def test_wanda_pruning_cpp_activation_weighting_differs_from_plain_magnitude():
    # One input feature (row of W, in the [K, N] layout) is scaled far above
    # the rest during calibration -- so its own small |W| entries can still
    # out-rank a larger-magnitude entry belonging to a quiet input channel.
    # This directly exercises the ||X_j||_2 multiplier: a plain-magnitude
    # pruning of the exact same weight would drop a different entry set.
    K, N = 8, 4
    salient_k = 2
    rng = np.random.default_rng(64)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    # Make salient_k's own row of W uniformly small in magnitude so it would
    # be dropped first under plain-|W| ranking...
    w[salient_k, :] = 0.02
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    x_cal = rng.standard_normal((64, K)).astype(np.float32)
    # ...but scale that same channel's activation up massively at
    # calibration time, so ||X_salient||_2 dominates every other channel's
    # own norm and Wanda should keep it despite its small |W|.
    x_cal[:, salient_k] *= 100.0

    wanda_pruned = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)

    w_wanda = _weight(wanda_pruned)
    w_magnitude = _magnitude_weight(magnitude_pruned)
    assert not np.array_equal(w_wanda, w_magnitude)
    # Plain magnitude drops the whole salient_k row (its own tied-smallest
    # entries); Wanda keeps at least one of them thanks to the activation
    # boost.
    assert np.count_nonzero(w_magnitude[salient_k, :]) == 0
    assert np.count_nonzero(w_wanda[salient_k, :]) > 0


# --- Conv: ordinary / depthwise / general grouped, all TRUE parity --------
#
# 2-D Conv is matched exactly like pruning.py's own `apply_wanda_pruning`
# used to -- ordinary (`group=1`), fully depthwise (`group == in_channels ==
# out_channels`), and general grouped (`1 < group < in_channels`) alike --
# via a from-scratch im2col per-`(in_channel, kh, kw)`-tap activation norm
# (ConvPatchSqSum) and a grouped/depthwise group-relative norm expansion
# (ConvGroupRelativeNorm), see structured_pruning_entry.h's own
# ApplyWandaPruning declaration comment. Since `apply_wanda_pruning` is now
# an alias of this port, these compare against a frozen golden fixture
# (see this file's own module docstring), same as every other test above.


def _conv_model(
    Cin, Cout, group, kh=3, kw=3, pads=(1, 1, 1, 1), strides=(1, 1), seed=0
):
    cin_per_group = Cin // group
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((Cout, cin_per_group, kh, kw)).astype(np.float32) * 0.4
    model = _model(
        f"""
        g (float[Nb,{Cin},H,W] X) => (float[Nb,{Cout},H2,W2] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], pads=[{pads[0]},{pads[1]},{pads[2]},{pads[3]}], strides=[{strides[0]},{strides[1]}], group={group}>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    return model, w


def test_wanda_pruning_cpp_conv_ordinary_group1_matches_python_reference():
    Cin, Cout = 6, 8
    model, w = _conv_model(Cin=Cin, Cout=Cout, group=1, seed=200)
    rng = np.random.default_rng(1200)
    x_cal = [rng.standard_normal((2, Cin, 10, 10)).astype(np.float32) for _ in range(2)]
    calib = [{"X": b} for b in x_cal]

    golden = _golden(_GOLDEN_CONV_ORDINARY_GROUP1)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=calib, sparsity=0.5
    )
    onnx.checker.check_model(actual)
    np.testing.assert_array_equal(_weight(actual), _weight(golden))
    assert not np.array_equal(_weight(actual), w)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_conv_fully_depthwise_matches_python_reference():
    Cin = Cout = group = 8
    model, w = _conv_model(Cin=Cin, Cout=Cout, group=group, seed=201)
    rng = np.random.default_rng(1201)
    x_cal = [rng.standard_normal((2, Cin, 10, 10)).astype(np.float32) for _ in range(2)]
    calib = [{"X": b} for b in x_cal]

    golden = _golden(_GOLDEN_CONV_FULLY_DEPTHWISE)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=calib, sparsity=0.5
    )
    onnx.checker.check_model(actual)
    np.testing.assert_array_equal(_weight(actual), _weight(golden))
    assert not np.array_equal(_weight(actual), w)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_conv_general_grouped_matches_python_reference():
    # 1 < group < in_channels -- exercises ConvGroupRelativeNorm's own
    # per-group channel-block slicing (not the group=1 or fully-depthwise
    # degenerate cases above).
    Cin, Cout, group = 8, 12, 4
    model, w = _conv_model(Cin=Cin, Cout=Cout, group=group, seed=202)
    rng = np.random.default_rng(1202)
    x_cal = [rng.standard_normal((2, Cin, 10, 10)).astype(np.float32) for _ in range(2)]
    calib = [{"X": b} for b in x_cal]

    golden = _golden(_GOLDEN_CONV_GENERAL_GROUPED)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=calib, sparsity=0.5
    )
    onnx.checker.check_model(actual)
    np.testing.assert_array_equal(_weight(actual), _weight(golden))
    assert not np.array_equal(_weight(actual), w)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_conv_general_grouped_nm_pattern_matches_python_reference():
    Cin, Cout, group = 8, 12, 4
    model, _w = _conv_model(Cin=Cin, Cout=Cout, group=group, seed=203)
    rng = np.random.default_rng(1203)
    x_cal = [rng.standard_normal((2, Cin, 10, 10)).astype(np.float32) for _ in range(2)]
    calib = [{"X": b} for b in x_cal]

    golden = _golden(_GOLDEN_CONV_GENERAL_GROUPED_NM)
    actual = onnxsim.apply_wanda_pruning_cpp(model, calibration_data=calib, n=2, m=4)
    onnx.checker.check_model(actual)
    np.testing.assert_array_equal(_weight(actual), _weight(golden))

    # Exactly 2 of every 4 consecutive columns survive, per output filter row
    # (the reshaped [out_channels, cin_per_group*kh*kw] view).
    cin_per_group = Cin // group
    w_nk = _weight(actual).reshape(Cout, cin_per_group * 3 * 3)
    for row in w_nk:
        for start in range(0, len(row), 4):
            group_cols = row[start : start + 4]
            if len(group_cols) == 4:
                assert np.count_nonzero(group_cols) == 2


def test_wanda_pruning_cpp_conv_depthwise_float16_matches_python_reference():
    # FLOAT16 analogue of the depthwise test above -- exercises the
    # ReadTensorAsF64/WriteF64TensorAs round trip for a Conv candidate, not
    # just the MatMul/Attention ones.
    Cin = Cout = group = 6
    rng = np.random.default_rng(204)
    w = (rng.standard_normal((Cout, 1, 3, 3)) * 0.4).astype(np.float16)
    model = _model(
        f"""
        g (float16[Nb,{Cin},10,10] X) => (float16[Nb,{Cout},10,10] Y)
        {{
          Y = Conv<kernel_shape=[3,3], pads=[1,1,1,1], group={group}>(X, W)
        }}
        """,
        initializer=[_f16(w, "W")],
    )
    rng2 = np.random.default_rng(1204)
    x_cal = [
        rng2.standard_normal((2, Cin, 10, 10)).astype(np.float16) for _ in range(2)
    ]
    calib = [{"X": b} for b in x_cal]

    golden = _golden(_GOLDEN_CONV_DEPTHWISE_FLOAT16)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=calib, sparsity=0.5
    )
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(golden).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w.view(np.uint16))
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


# --- FLOAT16/BFLOAT16 weight support: matches the pure-Python reference ----
#
# FLOAT16/BFLOAT16 is TRUE parity with pruning.py's own
# `apply_wanda_pruning` -- reads out upcast to float64 via
# ReadTensorAsF64, importance/masking computed identically to the FLOAT32
# path, written back down via WriteF64TensorAs, exactly mirroring
# pruning.py's own `_to_f64`/`_from_f64` round trip (see
# structured_pruning_entry.h's own ApplyWandaPruning declaration comment).


def test_wanda_pruning_cpp_matmul_float16_matches_python_reference():
    K, N = 16, 4
    rng = np.random.default_rng(61)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w, "W")],
    )
    onnx.checker.check_model(model)
    rng2 = np.random.default_rng(161)
    x_cal = rng2.standard_normal((16, K)).astype(np.float16)

    golden = _golden(_GOLDEN_MATMUL_FLOAT16)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    # Exact bit-pattern match against the Python reference (not
    # assert_allclose): both round-trip through float64 and mask, never
    # recompute, a surviving entry's own value.
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(golden).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w.view(np.uint16))
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_matmul_bfloat16_matches_python_reference():
    # onnxruntime has no BFLOAT16 CPU execution support in this environment
    # (confirmed separately: a plain BFLOAT16 MatMul session raises
    # NOT_IMPLEMENTED at session-creation time -- see this repo's own
    # test_magnitude_pruning_bfloat16_preserves_dtype_and_matches_array_
    # oracle for the same note) -- so calibration_data is deliberately `[]`
    # (not merely omitted -- omitting it triggers random calibration data
    # generation, still a real session) rather than a real batch: both
    # apply_wanda_pruning and apply_wanda_pruning_cpp skip calling the
    # executor entirely for zero calibration batches (see
    # WandaCalibrationStats' own top comment) and fall back to plain
    # per-layer magnitude importance, which is exactly what this test cross-
    # checks -- the BFLOAT16 candidate-matching/read-upcast/write-downcast
    # round trip, not the (here environment-unsupported) real activation
    # capture.
    K, N = 16, 4
    rng = np.random.default_rng(62)
    w = (rng.standard_normal((K, N)) * 0.5).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (bfloat16[batch,{K}] X) => (bfloat16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_bf16(w, "W")],
    )
    onnx.checker.check_model(model)

    golden = _golden(_GOLDEN_MATMUL_BFLOAT16)
    actual = onnxsim.apply_wanda_pruning_cpp(model, calibration_data=[], sparsity=0.5)
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.BFLOAT16
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(golden).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w.view(np.uint16))
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)


def test_wanda_pruning_cpp_attention_merged_qkv_float16_matches_python_reference():
    # FLOAT16 analogue of test_wanda_pruning_cpp_attention_merged_qkv_
    # matches_python_reference -- exercises MatchAttentionProducerWideDtype
    # (this pass' own local, dtype-widened copy of MatchAttentionProducer),
    # not just the MatMul/Gemm candidate path.
    hidden = 16
    nq = nk = nv = 8
    total_n = nq + nk + nv
    num_heads = 2
    rng = np.random.default_rng(63)
    w_qkv = (rng.standard_normal((hidden, total_n)) * 0.3).astype(np.float16)
    bias = (rng.standard_normal((total_n,)) * 0.05).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,seq,{hidden}] X) => (float16[batch,seq,{nv}] Y)
        {{
          Y, present = com.microsoft.Attention <num_heads={num_heads}>(X, Wqkv, Bqkv)
        }}
        """,
        initializer=[_f16(w_qkv, "Wqkv"), _f16(bias, "Bqkv")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    rng2 = np.random.default_rng(163)
    x_cal = rng2.standard_normal((3, 5, hidden)).astype(np.float16)

    golden = _golden(_GOLDEN_ATTENTION_MERGED_QKV_FLOAT16)
    actual = onnxsim.apply_wanda_pruning_cpp(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(actual)
    assert actual.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    np.testing.assert_array_equal(
        _weight(actual).view(np.uint16), _weight(golden).view(np.uint16)
    )
    assert not np.array_equal(_weight(actual).view(np.uint16), w_qkv.view(np.uint16))
    # Bias is never touched by unstructured/N:M pruning -- byte-identical.
    np.testing.assert_array_equal(_weight(actual, index=1), bias)
    assert onnxsim.weight_sparsity(actual) == pytest.approx(0.5, abs=0.1)
