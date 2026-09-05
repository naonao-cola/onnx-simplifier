"""Tests for ``onnxsim.apply_moe_whole_expert_pruning_cpp``/
``onnxsim.apply_qmoe_whole_expert_pruning_cpp`` -- the C++-backed ports of
``onnxsim.apply_moe_whole_expert_pruning``/``onnxsim.apply_qmoe_whole_expert_
pruning`` (see ``onnxsim/structured_pruning_entry.cpp``'s "MoE whole-expert
pruning"/"QMoE whole-expert pruning" section comments, and
``MoeRouterGateCalibrationStats``). Like
``tests/test_structured_wanda_pruning_cpp.py``, these run real calibration
data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor -- to capture per-expert mean router gate weight.

Unlike the already-ported ``apply_moe_expert_channel_pruning_cpp`` (which
narrows every expert's own ``inter_size`` identically, never touching
``num_experts``), this drops WHOLE experts: the ``num_experts`` leading axis
itself shrinks, together with the upstream router projection's own matching
output column -- see ``onnxsim/pruning.py``'s own "MoE whole-expert pruning"
section comment for the full masking-equivalence safety argument this port
carries over unchanged.

``onnxsim.apply_moe_whole_expert_pruning``/``onnxsim.apply_qmoe_whole_expert_
pruning`` (the pure-Python names) are now themselves thin aliases for
:func:`onnxsim.apply_moe_whole_expert_pruning_cpp`/:func:`onnxsim.apply_qmoe_
whole_expert_pruning_cpp` (full parity verified, including FLOAT16/BFLOAT16
weights and the router projection -- see pruning.py's own "MoE whole-expert
pruning"/"QMoE whole-expert pruning" section comments), so the handful of
tests below that used to call BOTH entry points and compare their live
outputs would be tautological (literally the same code path twice) if left
as-is. Those now instead compare the C++ port's output against a golden
fixture captured from the real pure-Python implementation *before* it was
deleted -- see ``_GOLDEN_*`` below (base64-encoded serialized ``ModelProto``
bytes, inlined directly per this repo's own established convention -- see
``tests/test_transformer_block_pruning_cpp.py``'s own identical precedent/
module-docstring for the full rationale).
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


# Frozen from onnxsim.apply_moe_whole_expert_pruning's/onnxsim.apply_qmoe_
# whole_expert_pruning's own real pure-Python implementations, on the exact
# model + calibration seed each corresponding test below builds, before
# those implementations were deleted in favor of the C++ port (see this
# file's own module docstring).
_GOLDEN_MOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE = (
    "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"
    "CkIECgAQEkIRCg1jb20ubWljcm9zb2Z0EAE="
)

_GOLDEN_MOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE_EMPTY_CALIBRATION = (
    "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"
    "CgAQEkIRCg1jb20ubWljcm9zb2Z0EAE="
)

_GOLDEN_QMOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE = (
    "CAo6hAgKHAoBWAoCUlcKAlJCEgFSGgZyb3V0ZXIiBEdlbW0KtAEKAVgKAVIKBEZDMVEKBEZDMVMK"
    "BEZDMUIKBEZDMlEKBEZDMlMKABIBWRoEcW1vZSIEUU1vRSoaCg9hY3RpdmF0aW9uX3R5cGUiBHJl"
    "bHWgAQMqGQoSZXhwZXJ0X3dlaWdodF9iaXRzGASgAQIqCAoBaxgCoAECKhQKCnF1YW50X3R5cGUi"
    "A2ludKABAyoXChB1c2Vfc3BhcnNlX21peGVyGACgAQI6DWNvbS5taWNyb3NvZnQSAWcqcAgECAYI"
    "BBACQgRGQzFRSmD3qGeIVmsMqveIlfCob4BN+j/JoPRFViBJX39Me/mTAAWIUIVKbQbrswq3+gaL"
    "yovxp3kWj/WOjCxfRoCtmd+2cP/G2feBaOr7uupK/tpwj117fPiEevlYEEh+c/9HvR8qbggECAYQ"
    "AUIERkMxU0pg5UvkPWYw4D1lIII92Q6WPa8pST0ASD49aJF2PeIJgD0YKts8PKiAPYbbGj27aac9"
    "5bkwPeGWkT3UNyU9UfiJPWmYRj33/Jo9onq0PRiqoj25ba49az+SPeG0Iz3ZfTA9KnAIBAgICAMQ"
    "AkIERkMyUUpgbYgL33tZ2M4ICWS99zBJWECrRQbPuQOEzYB+3fS6eHz1+EYUu0f597WTuPebvkoO"
    "wPQxc/Xtqmj49c35J/+MPGCf+ZyIl4uwkI6IBkBbvIYJAVrocKR7Nzuwwws4kbSPKo8BCAQICBAB"
    "QgRGQzJTSoAB7gSGPczEyj2xkSY9lPRzPUgaKT1WQbU9FmlwPaa3yT38cWM9Ub5QPZBilT2nAew8"
    "yfh5PacSmj11voU9eGNwPWhy7TzwYMI9WJKEPYbplj1+0KE97QufPbMonD1/arY9k5cqPfsFRz0d"
    "+Do9zRQ5PdQzlj1ZEx49VnBUPRdwTT0qjQEICAgEEAFCAlJXSoABApm4OxViYL2m2sw9Mf0mvhsO"
    "QD7giUk8cHWAvaLwYT6dLOW9WAIxvuU4lD1hYBm+M8m7vgo6j76zBiY9TyUtPkJI372AqTI9RGVb"
    "vkt6W750bLS9oQ2HvnoPob60E8K9yHhuvrFgNT6NQa+7IeBwvnqzCL5Hk5u8KOoPu3oTILsqGggE"
    "EAFCAlJCShARz68/zakWP9o6F0BWn3A/Km4IBAgGEAFCBEZDMUJKYC9Fjr7dasU+miRWP+fLCT6U"
    "ZcU/tlzWv+M/ob8asvQ+ZBeiv0Z3kj/vay8/9bwJP0rotz4mYaE+Yt24P/eytT/w2ws+4ldHP1su"
    "ALz9nMo+jzYMv2jNPr/2bj8+C92+v1oTCgFYEg4KDAgBEggKAggKCgIICGITCgFZEg4KDAgBEggK"
    "AggKCgIICEIECgAQEkIRCg1jb20ubWljcm9zb2Z0EAE="
)

_GOLDEN_QMOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE_EMPTY_CALIBRATION = (
    "CAo65gUKGgoBWAoCUlcSAVIaBnJvdXRlciIGTWF0TXVsCrABCgFYCgFSCgRGQzFRCgRGQzFTCgAK"
    "BEZDMlEKBEZDMlMKABIBWRoEcW1vZSIEUU1vRSoaCg9hY3RpdmF0aW9uX3R5cGUiBHJlbHWgAQMq"
    "GQoSZXhwZXJ0X3dlaWdodF9iaXRzGASgAQIqCAoBaxgBoAECKhQKCnF1YW50X3R5cGUiA2ludKAB"
    "AyoXChB1c2Vfc3BhcnNlX21peGVyGACgAQI6DWNvbS5taWNyb3NvZnQSAWcqWAgDCAYIBBACQgRG"
    "QzFRSkjirXYMg/iUVcP5s5eop478tptD9VlgdoQmtn99k/x5q5Q3D8HJT1+ebG8vNL5Ay6J1z8do"
    "k/+JZ5FIj7cFF0tVu2eo99+RV0AqVggDCAYQAUIERkMxU0pIrveTPaScSD0OOY49Oxy2PR45hz0/"
    "v5c9DzCnPbh6tz33XjQ9bKaMPZI1jj0cdRQ9OA/DPX+jhz3Zc1o96sXCPdZ7qT2inFo9KlgIAwgI"
    "CAMQAkIERkMyUUpIZxCJ+GTOjvSPH3sIvXP1dd0MCyBMvEegRWCz8g3ZeNDsy6UQA9ZF1h+ch/iD"
    "81xlTH9sjIz3BphXGiCQWAiGX3SfsKFcCFQrKm4IAwgIEAFCBEZDMlNKYMkCkT1CNlc91DWCPV7D"
    "hT3akps9meRKPa/1IT1hnBo92OefPax1Lj0iTgI9nbuSPWPjiz0NLZU91H/ZPTaGTj1r2Z89Zhdf"
    "Pf8mKT1NIzk99Y2CPeUUuz3q/o49AviCPSpsCAgIAxABQgJSV0pgLkAVvStDNL5SLdc8hzkDvmSs"
    "G7ydC7E9njCWPhJmXr4XEro9wmEhvkptNj2d1ag+dlRGvqo6jT7XMGs8eedWPensKD687kG8fUhO"
    "PnhUtT5Jr4m+zgxbPitOuT2APUW9WhMKAVgSDgoMCAESCAoCCAYKAggIYhMKAVkSDgoMCAESCAoC"
    "CAYKAggIQgQKABASQhEKDWNvbS5taWNyb3NvZnQQAQ=="
)

_GOLDEN_QMOE_WHOLE_EXPERT_BLOCKWISE_MATCHES_PYTHON_REFERENCE = (
    "CAo67xAKGgoBWAoCUlcSAVIaBnJvdXRlciIGTWF0TXVsCsMBCgFYCgFSCgRGQzFRCgRGQzFTCgAK"
    "BEZDMlEKBEZDMlMKABIBWRoEcW1vZSIEUU1vRSoaCg9hY3RpdmF0aW9uX3R5cGUiBHJlbHWgAQMq"
    "GQoSZXhwZXJ0X3dlaWdodF9iaXRzGASgAQIqCAoBaxgBoAECKhQKCnF1YW50X3R5cGUiA2ludKAB"
    "AyoXChB1c2Vfc3BhcnNlX21peGVyGACgAQIqEQoKYmxvY2tfc2l6ZRgQoAECOg1jb20ubWljcm9z"
    "b2Z0EgFnKpEECAIIEAgQEAJCBEZDMVFKgATwz2IX3BlsSFCqh3OWl36nH7JqyagIG+pYgJ1VI3l6"
    "dpMzr0OG2BjxiTZJ/U+dRYWARFZqiX5TrSxU9WFLUIxFSVlW9cgzdLg0y+u2+qa6sW/kdVuJu0Z/"
    "ZIB4Pq20iWWNqXlIf8tH9mzJCRGa5fX6i4uW92c4icMNe4ZoZSYpqXTIVWBYVoUiNaQo17XYf1/T"
    "CELEg1bWkYay+GSKsbmAV6CUd7/498d1sJtm1F4LOm/gf4I3hFpv5sePbZhlr5aJuHxxn5nUGF6q"
    "nFZvBI0EWIu8yzj3t7SFqlaPhEhGStqkek2gm1a9taYnVmypxp9baLaZpJm/U3H2N6h/M51lC7d8"
    "/avmuYedhNqaCZZpdJ9cR6U39GVcNbdoileYCraXplrZBplSj8+RlqBvAlZDb+hvpFeF+9hheEYo"
    "pG0JXEWnh66WYGlqRzk3Q6ngzPbo+JrK97h1vaNEB3GnjIuuq3n1mZY3amJsZ/t7pqOEyzj7mNqs"
    "WYhMSgjyybN6mfwWXPjEf7uI53CWhbhlWEbWtLT7uy14d8i8nP/IS7e3Yz8Jicu4iJlodpe2t0b0"
    "vJRzrU9ft5S9h5h52WepmVWfonec9qtoV5/4uTY1pGp7g42t63upe1KgfXL0TZ37I0yZuDDSRVtr"
    "lRxruqkKYSvbhH94Z6J6bbquC5Zs4DnOpiqRAggCCBAIAhABQgRGQzFTSoACKmqQPUyg+T2kclg9"
    "8NNsPc0Dlj0YEqw9guVgPV/8WT0ekcA98OfRPeKQjj06A689BuqTPaIneD0n/oA9tuqFPUTDzD0Z"
    "K4M9Y1quPf6Zjz1WbJo9e5CZPTHioD2GFcM97veiPZVZmj1A8dE9u0+bPQnEkD1JxN49BLK/PW4q"
    "jz2DzIY93PzZPS8RVz0UQrM973+9PVtlXT2wq8Y9fdSvPTIdyD2OOaY9cguoPfMdmz1uaq099G3J"
    "PXxjmj1ijZk9jg6oPT9Msz3jrnk9wPpxPTinBT6igYw9EGqGPf/Xxz3/rKA9g3nTPXWRnj2Yjoc9"
    "5ppsPaekkD0/2rI9cQ5KPSqRBAgCCCAICBACQgRGQzJRSoAE9pmsaZbFpox2OLcKpnc8bUlkFhVw"
    "g2SCq7T1W0m4PD5m1LtXtYRvEL6PYCdXuoulP5prt5ca8yfJ5xmfdJoHU3j31zrezKqpGlqmBGpz"
    "zDsXs5rnVI27BilWvKUKp472p4U1WyBnuXiY30/F2UNoqXxtN1aZawNpsZbct7WGCrpAy26Qf/WG"
    "jFCF/hiXg3917LOsVw+dhBq3Szv5r4l6affHbSaKY/lclbd1DFN4mVhMmLwKOQx0PrlqX3umx3Yr"
    "NQwmW5N/WWnHk752z7WJv1p+qH1/p3fEjfzAiTi4Z8aFZvhGXKfMJ4+5eYvCiok5mHiqf3zJl0dg"
    "EMRMoyZcTWGQb/WqjbhxWsmWlbafxoNv6cCT+3m3tIz2KW6Jxjq6t+XHD60VNYq22bmN1do2Cb34"
    "hmXC6nlKNXx6eovGaFC/ui5Fl+31gv54Vaa1e/aBqQhUVOujpQRQkEPOfKJ0FbIyl3i/KKRKL1kX"
    "MpRnDJ548yfJVLxXo0N+pKX92YaaiWh517jva2dpebuVb01WWra0RVhXcGBIVgukTk4zlInLqifN"
    "PwzjOgx6t3xJqparh5QIhc6kqGSpaqvIGIl4+H2JdbWZ+5WH5/Zkk7V2TIb7SXaE26CEyHYkl5yU"
    "j6lSCNmnhqH6kNn/13BkJdiB/GVn0I96TWPZ1ISJBehHKmcqkQIIAgggCAEQAUIERkMyU0qAApTV"
    "8T3Tj589CJuCPfc5oz2cWb49tJuHPRWYlT0KfqA9Ln+RPU+5pT04NIA9ED2rPdbeyT1KKcU9Z5al"
    "PXsMmz2brmQ9nDdRPYxApj1Odr49HOq4PdwovT3CtJY94a27Pet9iz1Vi6w9oi+zPYCfyT2/3pc9"
    "s7zSPTth4j2wHW49FylqPV0Moz1VSY09H4KAPXnorT02hLM9rLNsPXi4yD3teok9z0OCPULVtz3a"
    "HT49jtyMPSvyIT27ZIc90x25PVEr2j1097k96hu+Pc0LpD2mSHA9AkaxPVWHyD30T7I9dJm3PQXC"
    "vz2QXMY9Jk7YPbkMZz0P7Uo9eU82PbJ3oz0qjQIIIAgCEAFCAlJXSoACiPhWPnJZqj5bvIe7S/oI"
    "vq7PLr639jW+PfGKvGZJr71gHx69bnndvvvJ0j5jPxc+mQ4nPWGQTb1Apye+gfxCPdwxzr2gQIu+"
    "P+Z3voIZRD6Qgb8+WdHBPsNPzrtfqZm88Dg5PUHqBL2vA0K+0kzVuy+Ot739Voy9S3eyPlDzGb5G"
    "lF8+gBK+PaEFDD2R7XK+0hKGvVylQD6jFnu9V4a2vXUFhr1iJwQ9ihsjvk5dub04TbY9c7w9vmmz"
    "kD7sfes7R6iQvN37Hb5CeYW+dANOvkM/pr3puU6+Pv08Pncrsz0PFzS8N1saPS0mAj4CIqQ+ieA+"
    "voFBB75IoBq+rJFXPFoTCgFYEg4KDAgBEggKAggICgIIIGITCgFZEg4KDAgBEggKAggICgIIIEIE"
    "CgAQEkIRCg1jb20ubWljcm9zb2Z0EAE="
)


def _model(body, initializer=(), opset=18):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}, "com.microsoft": 1]
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


# Maps a numpy dtype (as passed to _moe_router_model's/_qmoe_router_model's
# own `float_dtype`) to the matching (ONNX text-format type name, ONNX enum,
# tensor-builder) triple -- lets both model builders build an
# activation-dtype-FLOAT16/BFLOAT16 model for the FP16/BFloat16 weight
# support tests below, while every existing FLOAT32 call site (the default)
# is completely unaffected.
_FLOAT_DTYPE_INFO = {
    np.dtype(np.float32): ("float", onnx.TensorProto.FLOAT, _f32),
    np.dtype(np.float16): ("float16", onnx.TensorProto.FLOAT16, _f16),
    np.dtype(ml_dtypes.bfloat16): ("bfloat16", onnx.TensorProto.BFLOAT16, _bf16),
}


def _u8(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.uint8), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


# --- Plain (float) MoE model builders ---------------------------------------


def _moe_router_model(
    fc1_w,
    fc2_w,
    router_w,
    router_b=None,
    fc1_b=None,
    k=2,
    tokens=6,
    use_sparse_mixer=0,
    float_dtype=np.float32,
):
    # `float_dtype` (FLOAT32 by default, every existing call site's own
    # behavior unchanged) controls X/R/Y's own activation dtype and every
    # FLOAT-family weight/bias's own storage dtype -- see the "FP16/BFloat16
    # weight support" section below, which passes FLOAT16/BFLOAT16 here to
    # build the widened-matcher (MatchMoeProducer AND MatchMoeRouterProducer)
    # test models.
    type_name, _, float_tensor = _FLOAT_DTYPE_INFO[np.dtype(float_dtype)]
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    router_call = "Gemm(X, RW, RB)" if router_b is not None else "Gemm(X, RW)"
    model = _model(
        f"""
        g ({type_name}[{tokens},{hidden}] X) => ({type_name}[{tokens},{hidden}] Y)
        {{
          R = {router_call}
          Y = com.microsoft.MoE <k={k}, activation_type="relu", use_sparse_mixer={use_sparse_mixer}> (X, R, FC1W, {fc1_b_arg}, FC2W)
        }}
        """
    )
    inits = [
        float_tensor(fc1_w, "FC1W"),
        float_tensor(fc2_w, "FC2W"),
        float_tensor(router_w, "RW"),
    ]
    if router_b is not None:
        inits.append(float_tensor(router_b, "RB"))
    if fc1_b is not None:
        inits.append(float_tensor(fc1_b, "FC1B"))
    model.graph.initializer.extend(inits)
    return model


def _moe_router_masking_oracle(
    fc1_w,
    fc2_w,
    router_w,
    router_b,
    dropped,
    k,
    fc1_b=None,
    tokens=6,
    float_dtype=np.float32,
):
    # Same-shape model with every `dropped` expert's routing logit forced to
    # a large negative value (Softmax assigns it ~0 probability) and its own
    # fc1/fc2 (+fc1_b) rows zeroed -- see onnxsim/pruning.py's own section
    # comment for why this is confirmed *exactly* equivalent to actually
    # removing the expert. -1e9 overflows to -inf in FLOAT16 (max magnitude
    # ~65504) -- -6e4 stays finite and representable while still forcing the
    # Softmax numerator to underflow to 0 given every other logit here is
    # `O(1)`.
    mask_value = -1e9 if np.dtype(float_dtype) == np.dtype(np.float32) else -6e4
    fc1_w_m = fc1_w.copy()
    fc2_w_m = fc2_w.copy()
    fc1_b_m = fc1_b.copy() if fc1_b is not None else None
    router_b_m = (
        router_b.copy()
        if router_b is not None
        else np.zeros(fc1_w.shape[0], float_dtype)
    )
    for e in dropped:
        fc1_w_m[e] = 0
        fc2_w_m[e] = 0
        if fc1_b_m is not None:
            fc1_b_m[e] = 0
        router_b_m[e] = mask_value
    return _moe_router_model(
        fc1_w_m,
        fc2_w_m,
        router_w,
        router_b=router_b_m,
        fc1_b=fc1_b_m,
        k=k,
        tokens=tokens,
        float_dtype=float_dtype,
    )


def _dropped_experts(router_w, kept_router_w):
    e = router_w.shape[1]
    kc = kept_router_w.shape[1]
    return [
        e_idx
        for e_idx in range(e)
        if not any(
            np.allclose(router_w[:, e_idx], kept_router_w[:, i]) for i in range(kc)
        )
    ]


# --- QMoE model builders (onnx.helper -- packed uint8 weights can't be
# expressed as onnx.parser text literals, see CLAUDE.md's own escape hatch
# for this case) ---------------------------------------------------------


def _qmoe_quantize_channel(w, bits):
    n, k = w.shape
    pack = 8 // bits
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    zp = 1 << (bits - 1)
    scale = np.abs(w).max(axis=1) / float(-qmin)
    scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), qmin, qmax).astype(np.int64) + zp
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8)
    parts = [(q[:, i::pack] & ((1 << bits) - 1)) for i in range(pack)]
    packed = np.zeros_like(parts[0])
    for i, p in enumerate(parts):
        packed = packed | (p << (bits * i))
    return packed.astype(np.uint8), scale


def _qmoe_quantize(w, bits):
    # Batched (per-expert) `_qmoe_quantize_channel`: w is [E, N, K].
    e, n, k = w.shape
    pack = 8 // bits
    packed = np.zeros((e, n, k // pack), dtype=np.uint8)
    scale = np.zeros((e, n), dtype=np.float32)
    for ei in range(e):
        packed[ei], scale[ei] = _qmoe_quantize_channel(w[ei], bits)
    return packed, scale


def _qmoe_router_model(
    fc1_q,
    fc1_scale,
    fc2_q,
    fc2_scale,
    bits,
    router_w,
    router_b=None,
    fc1_bias=None,
    k=2,
    tokens=6,
    use_sparse_mixer=0,
    float_dtype=np.float32,
):
    # `float_dtype` (FLOAT32 by default, every existing call site's own
    # behavior unchanged) controls X/R/Y's own activation dtype and every
    # FLOAT-family operand's own storage dtype (fc1/fc2 scale, fc1 bias,
    # router weight/bias) -- see the "FP16/BFloat16 weight support" section
    # below. `fc1_q`/`fc2_q` always stay UINT8 regardless.
    _, onnx_dtype, float_tensor = _FLOAT_DTYPE_INFO[np.dtype(float_dtype)]
    num_experts, inter, hidden_packed = fc1_q.shape
    hidden = hidden_packed * (8 // bits)
    inputs = [onnx.helper.make_tensor_value_info("X", onnx_dtype, [tokens, hidden])]
    outputs = [onnx.helper.make_tensor_value_info("Y", onnx_dtype, [tokens, hidden])]
    inits = [
        _u8(fc1_q, "FC1Q"),
        float_tensor(fc1_scale, "FC1S"),
        _u8(fc2_q, "FC2Q"),
        float_tensor(fc2_scale, "FC2S"),
        float_tensor(router_w, "RW"),
    ]
    if router_b is not None:
        router_node = onnx.helper.make_node(
            "Gemm", ["X", "RW", "RB"], ["R"], name="router"
        )
        inits.append(float_tensor(router_b, "RB"))
    else:
        router_node = onnx.helper.make_node("MatMul", ["X", "RW"], ["R"], name="router")

    node_inputs = ["X", "R", "FC1Q", "FC1S", "", "FC2Q", "FC2S", ""]
    if fc1_bias is not None:
        node_inputs[4] = "FC1B"
        inits.append(float_tensor(fc1_bias, "FC1B"))

    qmoe_node = onnx.helper.make_node(
        "QMoE",
        node_inputs,
        ["Y"],
        domain="com.microsoft",
        name="qmoe",
        k=k,
        activation_type="relu",
        expert_weight_bits=bits,
        quant_type="int",
        use_sparse_mixer=use_sparse_mixer,
    )
    graph = onnx.helper.make_graph(
        [router_node, qmoe_node], "g", inputs, outputs, initializer=inits
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def _qmoe_router_masking_oracle(
    fc1_q,
    fc1_scale,
    fc2_q,
    fc2_scale,
    bits,
    router_w,
    router_b,
    dropped,
    k,
    fc1_bias=None,
    tokens=6,
    float_dtype=np.float32,
):
    # See _moe_router_masking_oracle's own comment for why the mask value
    # differs by dtype (-1e9 overflows FLOAT16).
    mask_value = -1e9 if np.dtype(float_dtype) == np.dtype(np.float32) else -6e4
    fc1_q_m = fc1_q.copy()
    fc2_q_m = fc2_q.copy()
    fc1_bias_m = fc1_bias.copy() if fc1_bias is not None else None
    router_b_m = (
        router_b.copy()
        if router_b is not None
        else np.zeros(fc1_q.shape[0], float_dtype)
    )
    for e in dropped:
        fc1_q_m[e] = 0
        fc2_q_m[e] = 0
        if fc1_bias_m is not None:
            fc1_bias_m[e] = 0
        router_b_m[e] = mask_value
    return _qmoe_router_model(
        fc1_q_m,
        fc1_scale,
        fc2_q_m,
        fc2_scale,
        bits,
        router_w,
        router_b=router_b_m,
        fc1_bias=fc1_bias_m,
        k=k,
        tokens=tokens,
        float_dtype=float_dtype,
    )


# =============================================================================
# Plain (float) MoE
# =============================================================================


def test_moe_whole_expert_pruning_cpp_matches_ort_masking_oracle():
    E, hidden, inter, tokens = 5, 8, 6, 10
    rng = np.random.default_rng(101)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.4).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.4).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    k = 2
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, fc1_b=fc1_b, k=k, tokens=tokens
    )
    onnx.checker.check_model(model)

    calib_rng = np.random.default_rng(103)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model,
        calibration_data=calibration_data,
        sparsity=0.4,  # keep 3 of 5
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (3, inter, hidden)
    assert inits["FC2W"].shape == (3, hidden, inter)
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].shape == (3, inter)

    dropped = _dropped_experts(router_w, inits["RW"])
    assert len(dropped) == 2
    masked = _moe_router_masking_oracle(
        fc1_w, fc2_w, router_w, router_b, dropped, k, fc1_b=fc1_b, tokens=tokens
    )

    feed_rng = np.random.default_rng(107)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_moe_whole_expert_pruning_cpp_adversarial_low_usage_expert_dropped():
    # Expert 0's router bias is large and positive (dominant), the last
    # expert's is large and negative (rarely used) -- at sparsity=1/E (drop
    # exactly one), the low-usage expert must be the one dropped. Catches a
    # ranking bug that inverted the comparison or dropped the wrong expert.
    E, hidden, inter, tokens = 4, 6, 5, 8
    rng = np.random.default_rng(109)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float32)
    router_b = np.zeros(E, dtype=np.float32)
    router_b[0] = 8.0
    router_b[E - 1] = -8.0
    k = 1
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=k, tokens=tokens
    )

    calib_rng = np.random.default_rng(113)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=1.0 / E
    )
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (E - 1, inter, hidden)
    dropped = _dropped_experts(router_w, inits["RW"])
    assert dropped == [E - 1], f"expected the rarely-used expert dropped, got {dropped}"


def test_moe_whole_expert_pruning_cpp_k_is_floored_not_exceeded():
    # k=2 must never be pruned below -- requesting sparsity that would
    # remove more than num_experts - k experts is silently floored instead.
    E, hidden, inter, tokens = 5, 6, 4, 6
    rng = np.random.default_rng(127)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    k = 2
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.9
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (k, inter, hidden)
    assert inits["RW"].shape == (hidden, k)
    # Still a valid, executable model at the k floor.
    feed_rng = np.random.default_rng(131)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    _run(pruned, feeds)


def test_moe_whole_expert_pruning_cpp_zero_sparsity_is_a_no_op():
    E, hidden, inter, tokens = 4, 6, 4, 6
    rng = np.random.default_rng(140)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.0
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_moe_whole_expert_pruning_cpp_invalid_sparsity_raises():
    E, hidden, inter = 3, 6, 4
    rng = np.random.default_rng(141)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1)
    with pytest.raises(Exception):
        onnxsim.apply_moe_whole_expert_pruning_cpp(
            model, calibration_data=[], sparsity=1.0
        )


# --- Uncalibrated (empty calibration_data) weight-norm fallback ------------


def test_moe_whole_expert_pruning_cpp_empty_calibration_falls_back_to_weight_norm():
    E, hidden, inter, tokens = 5, 8, 6, 6
    rng = np.random.default_rng(150)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, fc1_b=fc1_b, k=1, tokens=tokens)

    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    kept = [
        e
        for e in range(E)
        if any(
            np.allclose(fc1_w[e], inits["FC1W"][i])
            for i in range(inits["FC1W"].shape[0])
        )
    ]

    expected_importance = np.sqrt(
        np.sum(np.square(fc1_w.astype(np.float64)), axis=(1, 2))
        + np.sum(np.square(fc2_w.astype(np.float64)), axis=(1, 2))
        + np.sum(np.square(fc1_b.astype(np.float64)), axis=1)
    )
    keep_count = inits["FC1W"].shape[0]
    expected_keep = sorted(np.argsort(-expected_importance)[:keep_count].tolist())
    assert sorted(kept) == expected_keep


# --- Cross-check against the pure-Python reference --------------------------


def test_moe_whole_expert_pruning_cpp_matches_python_reference():
    E, hidden, inter, tokens = 6, 10, 8, 12
    rng = np.random.default_rng(160)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, fc1_b=fc1_b, k=2, tokens=tokens
    )
    calib_rng = np.random.default_rng(161)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    golden = _golden(_GOLDEN_MOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE)
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()


def test_moe_whole_expert_pruning_cpp_matches_python_reference_empty_calibration():
    # sparsity=0.4 (not 0.5) deliberately avoids the exact n*sparsity == x.5
    # boundary, where Python's builtin `round()` (round-half-to-even) and
    # C++'s `std::llround` (round-half-away-from-zero) -- the SAME rounding
    # discrepancy every other already-ported chain family's own keep_count
    # computation in structured_pruning_entry.cpp carries (llround is this
    # codebase's established, pre-existing precedent, not something this
    # port introduces) -- can legitimately disagree by one kept expert.
    E, hidden, inter, tokens = 5, 8, 6, 6
    rng = np.random.default_rng(170)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)

    pruned_cpp = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    golden = _golden(
        _GOLDEN_MOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE_EMPTY_CALIBRATION
    )
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()


# --- Default (auto-generated) calibration data ------------------------------


def test_moe_whole_expert_pruning_cpp_default_calibration_data_runs():
    E, hidden, inter, tokens = 4, 6, 4, 6
    rng = np.random.default_rng(180)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, num_samples=3, seed=5, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].shape[0] == 2


# =============================================================================
# QMoE (quantized-weight MoE)
# =============================================================================


def test_qmoe_whole_expert_pruning_cpp_matches_ort_masking_oracle():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 10
    rng = np.random.default_rng(201)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    k = 2
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b=router_b,
        fc1_bias=fc1_b,
        k=k,
        tokens=tokens,
    )
    onnx.checker.check_model(model)

    calib_rng = np.random.default_rng(203)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1Q"].shape == (3, inter, hidden // 2)
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].shape == (3, inter)

    dropped = _dropped_experts(router_w, inits["RW"])
    assert len(dropped) == 2
    masked = _qmoe_router_masking_oracle(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b,
        dropped,
        k,
        fc1_bias=fc1_b,
        tokens=tokens,
    )
    onnx.checker.check_model(masked)

    feed_rng = np.random.default_rng(207)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.3}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_qmoe_whole_expert_pruning_cpp_k_is_floored_not_exceeded():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 6
    rng = np.random.default_rng(211)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    k = 2
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, k=k, tokens=tokens
    )
    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.9
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1Q"].shape == (k, inter, hidden // 2)
    assert inits["RW"].shape == (hidden, k)
    feed_rng = np.random.default_rng(213)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.3}
    _run(pruned, feeds)


# --- Uncalibrated (empty calibration_data) weight-norm fallback ------------


def test_qmoe_whole_expert_pruning_cpp_empty_calibration_falls_back_to_weight_norm():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 6
    rng = np.random.default_rng(221)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, fc1_bias=fc1_b, k=1, tokens=tokens
    )

    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    kept = [
        e
        for e in range(E)
        if any(
            np.array_equal(fc1_q[e], inits["FC1Q"][i])
            for i in range(inits["FC1Q"].shape[0])
        )
    ]

    # Dequantized weight-norm importance, mirroring
    # QMoEExpertWeightImportance/_qmoe_expert_weight_importance.
    def _dequant(q, s):
        e, n, kp = q.shape
        pack = 8 // bits
        parts = [(q >> (bits * i)) & ((1 << bits) - 1) for i in range(pack)]
        unpacked = np.stack(parts, axis=-1).reshape(e, n, kp * pack)
        return (unpacked.astype(np.float64) - (1 << (bits - 1))) * s[..., None].astype(
            np.float64
        )

    fc1_dq = _dequant(fc1_q, fc1_s)
    fc2_dq = _dequant(fc2_q, fc2_s)
    expected_importance = np.sqrt(
        np.sum(np.square(fc1_dq), axis=(1, 2))
        + np.sum(np.square(fc2_dq), axis=(1, 2))
        + np.sum(np.square(fc1_b.astype(np.float64)), axis=1)
    )
    keep_count = inits["FC1Q"].shape[0]
    expected_keep = sorted(np.argsort(-expected_importance)[:keep_count].tolist())
    assert sorted(kept) == expected_keep


# --- Cross-check against the pure-Python reference --------------------------


def test_qmoe_whole_expert_pruning_cpp_matches_python_reference():
    E, hidden, inter, bits, tokens = 6, 8, 6, 4, 10
    rng = np.random.default_rng(231)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b=router_b,
        fc1_bias=fc1_b,
        k=2,
        tokens=tokens,
    )
    calib_rng = np.random.default_rng(233)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    golden = _golden(_GOLDEN_QMOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE)
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()


def test_qmoe_whole_expert_pruning_cpp_matches_python_reference_empty_calibration():
    # sparsity=0.4 (not 0.5) -- see the plain-MoE analogue of this test's own
    # comment above for why the x.5 rounding boundary is deliberately
    # avoided here.
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 6
    rng = np.random.default_rng(241)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    fc1_q, fc1_s = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, k=1, tokens=tokens
    )

    pruned_cpp = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    golden = _golden(
        _GOLDEN_QMOE_WHOLE_EXPERT_MATCHES_PYTHON_REFERENCE_EMPTY_CALIBRATION
    )
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()


# --- Blockwise (block_size set) ---------------------------------------------


def test_qmoe_whole_expert_pruning_cpp_blockwise_matches_python_reference():
    # Confirms this pass needs no block_size-specific handling at all --
    # every per-expert tensor keeps num_experts as its own leading axis
    # regardless of block_size.
    E, hidden, inter, bits, block_size, tokens = 4, 32, 16, 4, 16, 8
    rng = np.random.default_rng(251)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)

    def _quantize_blockwise(w, bits, block_size):
        e, n, k = w.shape
        pack = 8 // bits
        kb = k // block_size
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        zp = 1 << (bits - 1)
        w_blocks = w.reshape(e, n, kb, block_size)
        scale = np.abs(w_blocks).max(axis=-1) / float(-qmin)
        scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
        q = (
            np.clip(np.round(w_blocks / scale[..., None]), qmin, qmax).astype(np.int64)
            + zp
        )
        q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8).reshape(e, n, k)
        parts = [(q[:, :, i::pack] & ((1 << bits) - 1)) for i in range(pack)]
        packed = np.zeros_like(parts[0])
        for i, p in enumerate(parts):
            packed = packed | (p << (bits * i))
        return packed.astype(np.uint8), scale

    fc1_q, fc1_s = _quantize_blockwise(fc1_w, bits, block_size)
    fc2_q, fc2_s = _quantize_blockwise(fc2_w, bits, block_size)
    model = _qmoe_router_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, router_w, k=1, tokens=tokens
    )
    # block_size isn't a _qmoe_router_model parameter -- patch the node
    # attribute directly (mirrors _qmoe_router_model's own block_size
    # kwarg in tests/test_pruning.py, kept out of this leaner helper).
    for node in model.graph.node:
        if node.op_type == "QMoE":
            node.attribute.append(onnx.helper.make_attribute("block_size", block_size))

    calib_rng = np.random.default_rng(253)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(3)
    ]
    pruned_cpp = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned_cpp)
    golden = _golden(_GOLDEN_QMOE_WHOLE_EXPERT_BLOCKWISE_MATCHES_PYTHON_REFERENCE)
    assert pruned_cpp.SerializeToString() == golden.SerializeToString()
    inits = _inits(pruned_cpp)
    assert inits["FC1Q"].shape[0] == inits["FC1S"].shape[0]


# --- Error handling ----------------------------------------------------------


def test_moe_whole_expert_pruning_cpp_missing_calibration_input_raises():
    E, hidden, inter = 3, 6, 4
    rng = np.random.default_rng(260)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1)
    bad_batch = {"NotX": np.zeros((2, hidden), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_moe_whole_expert_pruning_cpp(
            model, calibration_data=[bad_batch], sparsity=0.5
        )


# =============================================================================
# FP16/BFloat16 weight support
# =============================================================================
#
# MatchMoeWholeExpertProducer/MatchQMoEWholeExpertProducer (structured_
# pruning_entry.cpp) used to hard-require ``onnx.TensorProto.FLOAT`` for
# fc1/fc2's own weight/bias (via MatchMoeProducer/MatchQMoEProducer) AND for
# the router projection's own weight/bias (via the shared, FLOAT32-only
# MatchProducer) -- silently declining any node whose weights were stored as
# FLOAT16 or BFLOAT16, narrower than the pure-Python ``onnxsim.apply_moe_
# whole_expert_pruning``/``onnxsim.apply_qmoe_whole_expert_pruning``'s own
# ``_is_supported_float_dtype``. Now widened: fc1/fc2 via IsSupportedFloat
# Dtype/ReadTensorAsF64/WriteF64TensorAs (see structured_pruning_entry.cpp's
# own "MoE expert-intermediate-channel pruning" section top comment), and
# the router projection specifically via the new, narrowly-scoped
# MatchMoeRouterProducer/SliceMoeRouterWeight/SliceMoeRouterBias (see that
# function's own comment for why this is a dedicated local duplicate of the
# shared MatchProducer/SliceProducerWeight/SliceLastAxis rather than a
# change to those functions themselves).
#
# FLOAT16: confirmed separately that onnxruntime's CPU MoE/QMoE kernels
# execute genuine FLOAT16 end to end (matching test_moe_pruning_cpp.py's/
# test_qmoe_pruning_cpp.py's own identical notes), so these tests run real
# calibration through a real session, exactly like their FLOAT32 "matches_
# ort_masking_oracle" siblings above. BFLOAT16 has no onnxruntime CPU
# execution support in this environment at all -- so, since this pass
# ALWAYS needs a real executor to observe router activations for any
# non-empty ``calibration_data``, the BFLOAT16 tests below instead exercise
# the "no matching activation observed" weight-norm-only fallback path
# (``calibration_data=[]``, already covered for FLOAT32 by
# test_moe_whole_expert_pruning_cpp_empty_calibration_falls_back_to_weight_
# norm/its QMoE analogue) -- the executor is constructed but never actually
# `Run`, so no BFLOAT16 execution is ever attempted, while still exercising
# every matcher/slicer this section widened: fc1/fc2 weight+bias AND the
# router projection's own weight+bias.


def test_moe_whole_expert_pruning_cpp_fp16_matches_ort_masking_oracle():
    E, hidden, inter, tokens = 5, 8, 6, 10
    rng = np.random.default_rng(3001)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.4).astype(np.float16)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.4).astype(np.float16)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float16)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float16)
    router_b = rng.standard_normal(E).astype(np.float16)
    k = 2
    model = _moe_router_model(
        fc1_w,
        fc2_w,
        router_w,
        router_b=router_b,
        fc1_b=fc1_b,
        k=k,
        tokens=tokens,
        float_dtype=np.float16,
    )
    onnx.checker.check_model(model)
    inits_before = _inits(model)
    assert inits_before["FC1W"].dtype == np.float16
    assert inits_before["RW"].dtype == np.float16

    calib_rng = np.random.default_rng(3003)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float16)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model,
        calibration_data=calibration_data,
        sparsity=0.4,  # keep 3 of 5
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1W"].dtype == np.float16
    assert inits["FC1W"].shape == (3, inter, hidden)
    assert inits["FC2W"].shape == (3, hidden, inter)
    assert inits["RW"].dtype == np.float16
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].dtype == np.float16
    assert inits["FC1B"].shape == (3, inter)

    dropped = _dropped_experts(router_w, inits["RW"])
    assert len(dropped) == 2
    masked = _moe_router_masking_oracle(
        fc1_w,
        fc2_w,
        router_w,
        router_b,
        dropped,
        k,
        fc1_b=fc1_b,
        tokens=tokens,
        float_dtype=np.float16,
    )

    feed_rng = np.random.default_rng(3007)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float16)}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(
        out_pruned.astype(np.float64),
        out_masked.astype(np.float64),
        rtol=5e-2,
        atol=5e-2,
    )


def test_moe_whole_expert_pruning_cpp_fp16_adversarial_low_usage_expert_dropped():
    # FLOAT16 analogue of test_moe_whole_expert_pruning_cpp_adversarial_low_
    # usage_expert_dropped -- genuinely exercises the REAL calibration-based
    # mean-gate-weight ranking (MoeRouterGateCalibrationStats), not merely
    # the masking-equivalence safety property every other test above checks:
    # fc1/fc2 are plain random noise, uncorrelated with router usage, so a
    # bug that silently fell back to weight-norm-only importance for a
    # FLOAT16 model (MoeRouterGateCalibrationStats used to hard-require
    # FLOAT `router_probs`, dropping any FLOAT16 activation on the floor --
    # see that function's own comment) would drop an essentially random
    # expert here, not reliably the rarely-used one.
    E, hidden, inter, tokens = 4, 6, 5, 8
    rng = np.random.default_rng(3021)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float16)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float16)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float16)
    router_b = np.zeros(E, dtype=np.float16)
    router_b[0] = 8.0
    router_b[E - 1] = -8.0
    k = 1
    model = _moe_router_model(
        fc1_w,
        fc2_w,
        router_w,
        router_b=router_b,
        k=k,
        tokens=tokens,
        float_dtype=np.float16,
    )

    calib_rng = np.random.default_rng(3023)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float16)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=1.0 / E
    )
    inits = _inits(pruned)
    assert inits["FC1W"].shape == (E - 1, inter, hidden)
    dropped = _dropped_experts(router_w, inits["RW"])
    assert dropped == [E - 1], f"expected the rarely-used expert dropped, got {dropped}"


def test_moe_whole_expert_pruning_cpp_bfloat16_preserves_dtype_weight_norm_fallback():
    # No calibration data at all -- falls back to MoeExpertWeightImportance
    # (weight-norm-only ranking), so no BFLOAT16 execution is ever attempted
    # (see this section's own top comment). Still exercises the widened
    # fc1/fc2 weight+bias AND router weight+bias matcher/slicer end to end.
    E, hidden, inter, tokens = 4, 6, 5, 8
    rng = np.random.default_rng(3009)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(ml_dtypes.bfloat16)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(ml_dtypes.bfloat16)
    fc1_b = rng.standard_normal((E, inter)).astype(ml_dtypes.bfloat16)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(ml_dtypes.bfloat16)
    router_b = rng.standard_normal(E).astype(ml_dtypes.bfloat16)
    k = 1
    model = _moe_router_model(
        fc1_w,
        fc2_w,
        router_w,
        router_b=router_b,
        fc1_b=fc1_b,
        k=k,
        tokens=tokens,
        float_dtype=ml_dtypes.bfloat16,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_moe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    # floor = max(1, min(k, E)) = 1; keep_count = max(floor, E - round(E*0.4))
    # = max(1, 4 - 2) = 2.
    keep_count = 2
    assert inits["FC1W"].dtype == ml_dtypes.bfloat16
    assert inits["FC1W"].shape == (keep_count, inter, hidden)
    assert inits["RW"].dtype == ml_dtypes.bfloat16
    assert inits["RW"].shape == (hidden, keep_count)
    assert inits["FC1B"].dtype == ml_dtypes.bfloat16

    # Independent re-derivation of MoeExpertWeightImportance's own
    # weight-magnitude fallback ranking (root-sum-square over each expert's
    # own fc1/fc2(+fc1_b) slice), computed in float64 off the correctly-
    # decoded bfloat16 values.
    fc1_w64 = fc1_w.astype(np.float64)
    fc2_w64 = fc2_w.astype(np.float64)
    fc1_b64 = fc1_b.astype(np.float64)
    sq = (
        np.sum(fc1_w64**2, axis=(1, 2))
        + np.sum(fc2_w64**2, axis=(1, 2))
        + np.sum(fc1_b64**2, axis=1)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:keep_count])

    fc1_w_pruned = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "FC1W")
    )
    router_w_pruned = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "RW")
    )
    np.testing.assert_array_equal(
        fc1_w_pruned.view(np.uint16), fc1_w[keep].view(np.uint16)
    )
    np.testing.assert_array_equal(
        router_w_pruned.view(np.uint16), router_w[:, keep].view(np.uint16)
    )


def test_qmoe_whole_expert_pruning_cpp_fp16_matches_ort_masking_oracle():
    E, hidden, inter, bits, tokens = 5, 8, 6, 4, 10
    rng = np.random.default_rng(3011)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float16)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float16)
    router_b = rng.standard_normal(E).astype(np.float16)
    k = 2
    fc1_q, fc1_s32 = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s32 = _qmoe_quantize(fc2_w, bits)
    fc1_s = fc1_s32.astype(np.float16)
    fc2_s = fc2_s32.astype(np.float16)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b=router_b,
        fc1_bias=fc1_b,
        k=k,
        tokens=tokens,
        float_dtype=np.float16,
    )
    onnx.checker.check_model(model)
    inits_before = _inits(model)
    assert inits_before["FC1S"].dtype == np.float16
    assert inits_before["RW"].dtype == np.float16

    calib_rng = np.random.default_rng(3013)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float16)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    assert inits["FC1Q"].shape == (3, inter, hidden // 2)
    assert inits["FC1S"].dtype == np.float16
    assert inits["RW"].dtype == np.float16
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].dtype == np.float16

    dropped = _dropped_experts(router_w, inits["RW"])
    assert len(dropped) == 2
    masked = _qmoe_router_masking_oracle(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b,
        dropped,
        k,
        fc1_bias=fc1_b,
        tokens=tokens,
        float_dtype=np.float16,
    )
    onnx.checker.check_model(masked)

    feed_rng = np.random.default_rng(3017)
    feeds = {"X": (feed_rng.standard_normal((tokens, hidden)) * 0.3).astype(np.float16)}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(
        out_pruned.astype(np.float64),
        out_masked.astype(np.float64),
        rtol=5e-2,
        atol=5e-2,
    )


def test_qmoe_whole_expert_pruning_cpp_fp16_adversarial_low_usage_expert_dropped():
    # QMoE analogue of test_moe_whole_expert_pruning_cpp_fp16_adversarial_
    # low_usage_expert_dropped -- MoeRouterGateCalibrationStats is oblivious
    # to whether router_probs came from a MoE or QMoE node (see that
    # function's own comment), but this exercises the QMoE-specific match/
    # apply path (MatchQMoEWholeExpertProducer/ApplyQMoEWholeExpertChains)
    # end to end too, not just the shared calibration helper.
    E, hidden, inter, bits, tokens = 4, 6, 4, 4, 8
    rng = np.random.default_rng(3025)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float16)
    router_b = np.zeros(E, dtype=np.float16)
    router_b[0] = 8.0
    router_b[E - 1] = -8.0
    k = 1
    fc1_q, fc1_s32 = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s32 = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s32.astype(np.float16),
        fc2_q,
        fc2_s32.astype(np.float16),
        bits,
        router_w,
        router_b=router_b,
        k=k,
        tokens=tokens,
        float_dtype=np.float16,
    )

    calib_rng = np.random.default_rng(3027)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float16)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=1.0 / E
    )
    inits = _inits(pruned)
    assert inits["FC1Q"].shape == (E - 1, inter, hidden // 2)
    dropped = _dropped_experts(router_w, inits["RW"])
    assert dropped == [E - 1], f"expected the rarely-used expert dropped, got {dropped}"


def test_qmoe_whole_expert_pruning_cpp_bfloat16_preserves_dtype_weight_norm_fallback():
    # No calibration data at all -- falls back to QMoEExpertWeightImportance
    # (weight-norm-only ranking over the DEQUANTIZED weight, see that
    # function's own comment), so no BFLOAT16 execution is ever attempted.
    E, hidden, inter, bits, tokens = 4, 8, 6, 4, 8
    rng = np.random.default_rng(3019)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(ml_dtypes.bfloat16)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(ml_dtypes.bfloat16)
    router_b = rng.standard_normal(E).astype(ml_dtypes.bfloat16)
    k = 1
    fc1_q, fc1_s32 = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s32 = _qmoe_quantize(fc2_w, bits)
    fc1_s = fc1_s32.astype(ml_dtypes.bfloat16)
    fc2_s = fc2_s32.astype(ml_dtypes.bfloat16)
    model = _qmoe_router_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        router_w,
        router_b=router_b,
        fc1_bias=fc1_b,
        k=k,
        tokens=tokens,
        float_dtype=ml_dtypes.bfloat16,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_whole_expert_pruning_cpp(
        model, calibration_data=[], sparsity=0.4
    )
    onnx.checker.check_model(pruned)
    inits = _inits(pruned)
    # floor = max(1, min(k, E)) = 1; keep_count = max(floor, E - round(E*0.4))
    # = max(1, 4 - 2) = 2.
    keep_count = 2
    assert inits["FC1Q"].shape == (keep_count, inter, hidden // 2)
    assert inits["FC1S"].dtype == ml_dtypes.bfloat16
    assert inits["RW"].dtype == ml_dtypes.bfloat16
    assert inits["RW"].shape == (hidden, keep_count)
    assert inits["FC1B"].dtype == ml_dtypes.bfloat16

    # Independent re-derivation of QMoEExpertWeightImportance's own
    # weight-magnitude fallback ranking: dequantize with the ACTUAL
    # bfloat16-rounded scale/bias this model stores (not the internal
    # float32 quantizer scale), matching what the C++ port itself reads
    # back via ReadTensorAsF64. Mirrors this file's own inline `_dequant`
    # helper in test_qmoe_whole_expert_pruning_cpp_empty_calibration_falls_
    # back_to_weight_norm above.
    def _dequant(q, s):
        e, n, kp = q.shape
        pack = 8 // bits
        parts = [(q >> (bits * i)) & ((1 << bits) - 1) for i in range(pack)]
        unpacked = np.stack(parts, axis=-1).reshape(e, n, kp * pack)
        return (unpacked.astype(np.float64) - (1 << (bits - 1))) * s[..., None].astype(
            np.float64
        )

    fc1_dq = _dequant(fc1_q, fc1_s)
    fc2_dq = _dequant(fc2_q, fc2_s)
    fc1_b64 = fc1_b.astype(np.float64)
    sq = (
        np.sum(fc1_dq**2, axis=(1, 2))
        + np.sum(fc2_dq**2, axis=(1, 2))
        + np.sum(fc1_b64**2, axis=1)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:keep_count])

    fc1_q_pruned = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "FC1Q")
    )
    router_w_pruned = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "RW")
    )
    np.testing.assert_array_equal(fc1_q_pruned, fc1_q[keep])
    np.testing.assert_array_equal(
        router_w_pruned.view(np.uint16), router_w[:, keep].view(np.uint16)
    )
