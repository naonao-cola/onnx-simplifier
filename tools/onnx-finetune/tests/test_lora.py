"""End-to-end tests for the LoRA/QLoRA scripts under tools/onnx-finetune/
scripts: pure graph-surgery correctness, run via subprocess exactly as a
real user would invoke each CLI. Needs only onnx + numpy (+ pytest) --
unlike generate_artifacts.py/onnx-finetune itself, which need a
training-enabled onnxruntime build (see ../README.md), these scripts never
touch onnxruntime.

Correctness is checked with onnx.reference.ReferenceEvaluator (bundled with
the onnx package, no onnxruntime needed): every injected adapter must be a
numerical no-op at its zero/Kaiming-normal init, and a "trained" adapter
(lora_B perturbed away from zero, standing in for a real training run)
extracted and re-applied to a fresh copy of the base model must reproduce
the fine-tuned model's output exactly.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True)


def _perturb_lora_b(model_path, out_path, seed=7, scale=0.05):
    model = onnx.load(model_path)
    rng = np.random.default_rng(seed)
    for init in model.graph.initializer:
        if init.name.endswith(".lora_B"):
            arr = numpy_helper.to_array(init)
            new = (rng.standard_normal(arr.shape) * scale).astype(arr.dtype)
            init.CopyFrom(numpy_helper.from_array(new, init.name))
    onnx.save(model, out_path)


def _matmul_model(path, in_dim=4, hidden=16, out_dim=1, seed=0):
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((in_dim, hidden)) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((hidden, out_dim)) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", ["input", "fc1.weight"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("MatMul", ["r", "fc2.weight"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "matmul_net",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, in_dim])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, out_dim])],
        initializer=[
            numpy_helper.from_array(w1, "fc1.weight"),
            numpy_helper.from_array(w2, "fc2.weight"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _gemm_model(path, trans_a, trans_b, in_dim=5, out_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    w_shape = (out_dim, in_dim) if trans_b else (in_dim, out_dim)
    w = (rng.standard_normal(w_shape) * 0.1).astype(np.float32)
    b = np.zeros(out_dim, dtype=np.float32)
    x_shape = [in_dim, None] if trans_a else [None, in_dim]
    x_name = "inputT" if trans_a else "input"
    node = helper.make_node(
        "Gemm",
        [x_name, "fc.weight", "fc.bias"],
        ["output"],
        transA=int(trans_a),
        transB=int(trans_b),
    )
    graph = helper.make_graph(
        [node],
        "gemm_net",
        [helper.make_tensor_value_info(x_name, TensorProto.FLOAT, x_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, out_dim])],
        initializer=[
            numpy_helper.from_array(w, "fc.weight"),
            numpy_helper.from_array(b, "fc.bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return x_name


def _conv1x1_model(path, in_ch=3, mid_ch=8, out_ch=4, stride=2, seed=0):
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((mid_ch, in_ch, 1, 1)) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((out_ch, mid_ch, 1, 1)) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv1.weight"],
            ["h"],
            kernel_shape=[1, 1],
            strides=[stride, stride],
        ),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node(
            "Conv", ["r", "conv2.weight"], ["output"], kernel_shape=[1, 1]
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "conv1x1_net",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [None, in_ch, 8, 8]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [None, out_ch, None, None]
            )
        ],
        initializer=[
            numpy_helper.from_array(w1, "conv1.weight"),
            numpy_helper.from_array(w2, "conv2.weight"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _assert_noop_at_init(base_path, lora_path, feeds):
    y0 = ReferenceEvaluator(str(base_path)).run(None, feeds)[0]
    y1 = ReferenceEvaluator(str(lora_path)).run(None, feeds)[0]
    assert np.allclose(y0, y1, atol=1e-6), np.abs(y0 - y1).max()


def _assert_round_trips(tmp_path, base_path, lora_path, manifest_path, feeds):
    finetuned_path = tmp_path / "finetuned.onnx"
    _perturb_lora_b(lora_path, finetuned_path)

    adapter_path = tmp_path / "adapter.onnx"
    _run(
        "extract_lora_adapter.py",
        str(finetuned_path),
        "--params-file",
        str(manifest_path),
        "-o",
        str(adapter_path),
    )

    applied_path = tmp_path / "applied.onnx"
    _run(
        "apply_lora_adapter.py",
        str(base_path),
        "--adapter",
        str(adapter_path),
        "--params-file",
        str(manifest_path),
        "-o",
        str(applied_path),
    )

    y_ft = ReferenceEvaluator(str(finetuned_path)).run(None, feeds)[0]
    y_applied = ReferenceEvaluator(str(applied_path)).run(None, feeds)[0]
    y_base = ReferenceEvaluator(str(base_path)).run(None, feeds)[0]
    assert np.allclose(y_ft, y_applied, atol=1e-6)
    assert np.abs(y_base - y_applied).max() > 1e-3  # the adapter did something


def test_matmul_injection_noop_at_init_and_round_trips(tmp_path):
    base_path = tmp_path / "base.onnx"
    _matmul_model(base_path)
    lora_path = tmp_path / "lora.onnx"
    manifest_path = tmp_path / "manifest.json"
    _run(
        "inject_lora.py",
        str(base_path),
        "-o",
        str(lora_path),
        "--rank",
        "2",
        "--params-out",
        str(manifest_path),
    )

    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["pairs"]) == 2  # fc1.weight, fc2.weight

    x = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    _assert_noop_at_init(base_path, lora_path, {"input": x})
    _assert_round_trips(tmp_path, base_path, lora_path, manifest_path, {"input": x})


def test_gemm_transb_injection_noop_at_init(tmp_path):
    base_path = tmp_path / "base.onnx"
    _gemm_model(base_path, trans_a=False, trans_b=True)
    lora_path = tmp_path / "lora.onnx"
    manifest_path = tmp_path / "manifest.json"
    _run(
        "inject_lora.py",
        str(base_path),
        "-o",
        str(lora_path),
        "--rank",
        "2",
        "--params-out",
        str(manifest_path),
    )
    x = np.random.default_rng(2).standard_normal((6, 5)).astype(np.float32)
    _assert_noop_at_init(base_path, lora_path, {"input": x})
    _assert_round_trips(tmp_path, base_path, lora_path, manifest_path, {"input": x})


def test_gemm_transa_injection_noop_at_init(tmp_path):
    base_path = tmp_path / "base.onnx"
    _gemm_model(base_path, trans_a=True, trans_b=True)
    lora_path = tmp_path / "lora.onnx"
    manifest_path = tmp_path / "manifest.json"
    _run(
        "inject_lora.py",
        str(base_path),
        "-o",
        str(lora_path),
        "--rank",
        "2",
        "--params-out",
        str(manifest_path),
    )
    xt = np.random.default_rng(3).standard_normal((5, 6)).astype(np.float32)
    _assert_noop_at_init(base_path, lora_path, {"inputT": xt})
    _assert_round_trips(tmp_path, base_path, lora_path, manifest_path, {"inputT": xt})


def test_conv1x1_injection_noop_at_init(tmp_path):
    base_path = tmp_path / "base.onnx"
    _conv1x1_model(base_path)
    lora_path = tmp_path / "lora.onnx"
    manifest_path = tmp_path / "manifest.json"
    _run(
        "inject_lora.py",
        str(base_path),
        "-o",
        str(lora_path),
        "--rank",
        "2",
        "--params-out",
        str(manifest_path),
    )
    x = np.random.default_rng(4).standard_normal((2, 3, 8, 8)).astype(np.float32)
    _assert_noop_at_init(base_path, lora_path, {"input": x})
    _assert_round_trips(tmp_path, base_path, lora_path, manifest_path, {"input": x})


def test_target_contains_filters_by_name(tmp_path):
    base_path = tmp_path / "base.onnx"
    _matmul_model(base_path)
    lora_path = tmp_path / "lora.onnx"
    manifest_path = tmp_path / "manifest.json"
    _run(
        "inject_lora.py",
        str(base_path),
        "-o",
        str(lora_path),
        "--rank",
        "2",
        "--target-contains",
        "fc1.weight",
        "--params-out",
        str(manifest_path),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["pairs"] == [["fc1.weight.lora_A", "fc1.weight.lora_B"]]


def test_prepare_qlora_excludes_adapter_from_quantization(tmp_path):
    pytest.importorskip("onnxsim.nf4")
    base_path = tmp_path / "base.onnx"
    _matmul_model(base_path, in_dim=64, hidden=128, out_dim=8)
    qlora_path = tmp_path / "qlora.onnx"
    manifest_path = tmp_path / "manifest.json"
    _run(
        "prepare_qlora.py",
        str(base_path),
        "-o",
        str(qlora_path),
        "--rank",
        "2",
        "--block-size",
        "64",
        "--params-out",
        str(manifest_path),
    )

    model = onnx.load(qlora_path)
    names = {t.name for t in model.graph.initializer}
    manifest = json.loads(manifest_path.read_text())
    for a_name, b_name in manifest["pairs"]:
        assert f"{a_name}_nf4_q" not in names
        assert f"{b_name}_nf4_q" not in names
    assert any(n.endswith("_nf4_q") for n in names)  # the base weights WERE quantized

    x = np.random.default_rng(5).standard_normal((4, 64)).astype(np.float32)
    y_base = ReferenceEvaluator(str(base_path)).run(None, {"input": x})[0]
    y_qlora = ReferenceEvaluator(str(qlora_path)).run(None, {"input": x})[0]
    rel_l2 = np.linalg.norm(y_base - y_qlora) / np.linalg.norm(y_base)
    assert rel_l2 < 0.25  # NF4 quantization noise only, adapters are zero-init
