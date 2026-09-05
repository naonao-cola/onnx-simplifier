"""Real-hardware follow-up to the Conv/MatMul-specific coverage tests:
widens out to the full 92-op `AX650_SUPPORTED_OPS` list. Compiling a
single-node-per-op battery through a real `pulsar2 build` took confirmed
coverage from 27/92 (~29%) to 91/92 (~99%) -- see scripts/axera/README.md's
"Systematic op coverage: from 29% to 99% of AX650_SUPPORTED_OPS" section
for the full narrative and the complete pass/fail list.

This file locks in only the most valuable, generalizable, or surprising
findings from that sweep as real regression tests -- not all 91 confirmed
ops, which would be a lot of low-value maintenance for simple ops unlikely
to regress. In particular: a real, generalizable gotcha (Pulsar2 doesn't
resolve an ONNX attribute's schema default -- it must be set explicitly on
the node or the build fails with a confusing internal error, not a graceful
"unsupported" one), a native op newly confirmed useful for future onnxsim
reconstruction work (RMSNormalization), and a few confirmed-failing ops
despite being listed in AX650_SUPPORTED_OPS.

Needs a loaded `pulsar2:*` Docker image -- skip-guarded like
tests/test_pulsar2_hf_to_axmodel.py.
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(),
    reason=f"pulsar2 Docker image not loaded: {pulsar2_docker.DEFAULT_IMAGE}",
)


def _const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _single_node_model(
    op_type,
    node_inputs,
    node_outputs,
    out_shapes,
    initializers=(),
    attrs=None,
    opset=17,
):
    attrs = attrs or {}
    node = helper.make_node(op_type, node_inputs, node_outputs, **attrs)
    graph = helper.make_graph(
        [node],
        f"g_{op_type}",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, list(sh))
            for o, sh in zip(node_outputs, out_shapes)
        ],
        initializer=list(initializers),
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _build(tmp_path, name, model):
    work_dir = os.path.join(str(tmp_path), name)
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    samples = [rng.standard_normal((1, 4)).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib_x.tar"), samples
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    import json

    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(cfg, f)
    return pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )


@pytest.mark.parametrize(
    "op_type,default_attrs,explicit_attrs",
    [
        ("Elu", {}, {"alpha": 1.0}),
        ("LeakyRelu", {}, {"alpha": 0.01}),
    ],
)
def test_unset_default_attribute_crashes_but_explicit_value_compiles(
    tmp_path, op_type, default_attrs, explicit_attrs
):
    """Confirmed real, generalizable gotcha: Pulsar2's frontend does not
    resolve an ONNX attribute's own schema default when the attribute is
    left unset on the node -- it reads it as None and crashes with a
    confusing internal error, not a graceful 'unsupported' one. Setting
    the exact same default value explicitly fixes it."""
    model_default = _single_node_model(
        op_type, ["x"], ["y"], [(1, 4)], attrs=default_attrs
    )
    result_default = _build(tmp_path, f"{op_type}_default", model_default)
    assert not result_default.success

    model_explicit = _single_node_model(
        op_type, ["x"], ["y"], [(1, 4)], attrs=explicit_attrs
    )
    result_explicit = _build(tmp_path, f"{op_type}_explicit", model_explicit)
    assert result_explicit.success, result_explicit.error


def test_topk_unset_largest_sorted_crashes_but_explicit_compiles(tmp_path):
    k_init = _const("k", np.array([2], dtype=np.int64))

    model_default = _single_node_model(
        "TopK",
        ["x", "k"],
        ["values", "indices"],
        [(1, 2), (1, 2)],
        initializers=[k_init],
        attrs={"axis": 1},
    )
    # values/indices have different dtypes; patch indices output dtype
    model_default.graph.output[1].type.tensor_type.elem_type = onnx.TensorProto.INT64
    result_default = _build(tmp_path, "topk_default", model_default)
    assert not result_default.success

    model_explicit = _single_node_model(
        "TopK",
        ["x", "k"],
        ["values", "indices"],
        [(1, 2), (1, 2)],
        initializers=[k_init],
        attrs={"axis": 1, "largest": 1, "sorted": 1},
    )
    model_explicit.graph.output[1].type.tensor_type.elem_type = onnx.TensorProto.INT64
    result_explicit = _build(tmp_path, "topk_explicit", model_explicit)
    assert result_explicit.success, result_explicit.error


def test_rms_normalization_native_op_compiles(tmp_path):
    """Confirmed real: the native ONNX RMSNormalization op (opset 23)
    compiles successfully as a single op -- reconstruct_hf_graph()
    currently hand-decomposes RMSNorm into ReduceMean/Add/Sqrt/Div/Mul
    instead (written before this op existed in the ONNX opset)."""
    model = _single_node_model(
        "RMSNormalization",
        ["x", "scale"],
        ["y"],
        [(1, 4)],
        initializers=[_const("scale", np.ones(4, dtype=np.float32))],
        attrs={"axis": -1, "epsilon": 1e-5},
        opset=23,
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    result = _build(tmp_path, "rms_norm", model)
    assert result.success, result.error


def test_silu_nonstandard_op_compiles(tmp_path):
    """Confirmed real: Silu isn't a real ONNX operator schema at all (no
    onnx.checker validation possible), but it's a real, working Axera
    extension op name -- Pulsar2 recognizes it directly. onnxsim should
    still keep emitting the standard Sigmoid+Mul decomposition for
    onnxruntime compatibility; this just confirms the raw op works."""
    model = _single_node_model("Silu", ["x"], ["y"], [(1, 4)])
    result = _build(tmp_path, "silu_raw", model)
    assert result.success, result.error


@pytest.mark.parametrize("op_type", ["Xor", "Swish"])
def test_listed_op_fails_as_unimplemented(tmp_path, op_type):
    """Confirmed real: these are listed in AX650_SUPPORTED_OPS but fail
    outright on a real build -- a real, confirmed gap between the
    doc-scraped list and what actually compiles."""
    if op_type == "Xor":
        model = _single_node_model(
            "Xor",
            ["x", "c"],
            ["y"],
            [(1, 4)],
            initializers=[_const("c", np.ones((4,), dtype=bool))],
        )
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.BOOL
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.BOOL
    else:
        model = _single_node_model(
            "Swish", ["x"], ["y"], [(1, 4)], attrs={"alpha": 1.0}, opset=24
        )
        model.ir_version = 10

    result = _build(tmp_path, f"{op_type.lower()}_fails", model)
    assert not result.success


def test_rotary_embedding_fails_even_with_explicit_attributes(tmp_path):
    """Confirmed real: unlike Elu/LeakyRelu/TopK above, setting
    RotaryEmbedding's optional attributes explicitly (interleaved=0,
    rotary_embedding_dim=0) does NOT fix it -- genuinely unimplemented,
    not an attribute-defaulting issue this time.
    reconstruct_hf_graph()'s hand-decomposed RoPE remains the only working
    path for real hardware."""
    batch, num_heads, seq_len, head_dim = 1, 1, 8, 8
    cos_cache = np.ones((seq_len, head_dim // 2), dtype=np.float32)
    sin_cache = np.zeros((seq_len, head_dim // 2), dtype=np.float32)
    node = helper.make_node(
        "RotaryEmbedding",
        ["x", "cos_cache", "sin_cache"],
        ["y"],
        num_heads=num_heads,
        interleaved=0,
        rotary_embedding_dim=0,
    )
    graph = helper.make_graph(
        [node],
        "g_rope",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [batch, num_heads, seq_len, head_dim]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, [batch, num_heads, seq_len, head_dim]
            )
        ],
        initializer=[_const("cos_cache", cos_cache), _const("sin_cache", sin_cache)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 23)], ir_version=10
    )
    onnx.checker.check_model(model)

    work_dir = tmp_path / "rope_work"
    work_dir.mkdir()
    onnx.save(model, str(work_dir / "model.onnx"))
    rng = np.random.default_rng(0)
    (work_dir / "dataset").mkdir()
    samples = [
        rng.standard_normal((batch, num_heads, seq_len, head_dim)).astype(np.float32)
        for _ in range(4)
    ]
    pulsar2_docker.make_numpy_calibration_tar(
        str(work_dir / "dataset" / "calib_x.tar"), samples
    )
    import json

    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    (work_dir / "config").mkdir()
    with open(work_dir / "config" / "cfg.json", "w") as f:
        json.dump(cfg, f)

    result = pulsar2_docker.build(
        str(work_dir), "model.onnx", "output", config_path="config/cfg.json"
    )
    assert not result.success
