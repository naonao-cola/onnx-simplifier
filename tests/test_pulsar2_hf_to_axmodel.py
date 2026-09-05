"""Integration test for the hf-config+safetensors -> onnx -> axmodel path:
``onnxsim.reconstruct_hf_graph()`` feeding a real ``pulsar2 build`` (Docker)
via ``scripts/axera/pulsar2_docker.build_from_hf_checkpoint()``.

Needs a loaded ``pulsar2:*`` Docker image (see ``pulsar2_docker.py``'s
module docstring for how to get one) -- not available in ordinary CI, so
this is skip-guarded like the dormant ``pulsar2-docker-convert`` job in
``.github/workflows/axera-integration.yml``. Confirmed real end to end in
the session that added this test: a synthetic tiny Llama-shaped checkpoint
compiled to a `compiled.axmodel` and ran successfully on a real AX650N.
"""

import json
import os
import struct
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402
import pulsar2_ops  # noqa: E402

import onnxsim  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(),
    reason=f"pulsar2 Docker image not loaded: {pulsar2_docker.DEFAULT_IMAGE}",
)


def _write_tiny_llama_checkpoint(hf_dir, bf16=False):
    rng = np.random.default_rng(0)
    n_embd, n_head, n_layer, n_ff, vocab = 16, 2, 2, 32, 32

    def rand(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.1).astype("<f4")

    weights = {"model.embed_tokens.weight": rand(vocab, n_embd)}
    for i in range(n_layer):
        p = f"model.layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.self_attn.q_proj.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.self_attn.k_proj.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.self_attn.v_proj.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.self_attn.o_proj.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.post_attention_layernorm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.mlp.gate_proj.weight"] = rand(n_ff, n_embd)
        weights[f"{p}.mlp.up_proj.weight"] = rand(n_ff, n_embd)
        weights[f"{p}.mlp.down_proj.weight"] = rand(n_embd, n_ff)
    weights["model.norm.weight"] = rand(n_embd) + 1.0

    config = {
        "model_type": "llama",
        "hidden_size": n_embd,
        "num_hidden_layers": n_layer,
        "intermediate_size": n_ff,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        "tie_word_embeddings": True,
        "attention_bias": False,
    }
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(config, f)

    header = {}
    offset = 0
    blobs = []
    for name, arr in weights.items():
        if bf16:
            # truncate to the top 16 bits of each float32 (exact bf16 cast)
            u16 = (arr.view("<u4") >> 16).astype("<u2")
            data = u16.tobytes()
            dtype = "BF16"
        else:
            data = arr.tobytes()
            dtype = "F32"
        nbytes = len(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        blobs.append(data)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(os.path.join(hf_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def test_hf_checkpoint_reconstructed_onnx_compiles_to_a_real_axmodel(tmp_path):
    hf_dir = tmp_path / "tiny_llama"
    hf_dir.mkdir()
    _write_tiny_llama_checkpoint(str(hf_dir))

    # Independently confirm reconstruct_hf_graph()'s own output is valid
    # before handing it to Docker -- a failure here is onnxsim's bug, not
    # Pulsar2's.
    model = onnxsim.reconstruct_hf_graph(str(hf_dir), batch_size=1, seq_len=8)
    onnx.checker.check_model(model)

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    result = pulsar2_docker.build_from_hf_checkpoint(
        str(hf_dir), str(work_dir), "output"
    )

    assert result.success, result.error
    assert result.axmodel_path is not None
    assert os.path.exists(result.axmodel_path)

    expected_phases = {
        "docker_check",
        "import_onnxsim",
        "reconstruct_onnx",
        "calibration_data",
        "pulsar2_build",
    }
    assert expected_phases <= result.phase_timings.keys()
    assert all(v >= 0 for v in result.phase_timings.values())
    assert result.phase_timings["total"] == pytest.approx(
        sum(v for k, v in result.phase_timings.items() if k != "total"), rel=1e-6
    )

    compiled = onnx.load(result.axmodel_path)
    op_types = {n.op_type for n in compiled.graph.node}
    assert pulsar2_ops.AXERA_NPU_OP_TYPE in op_types
    assert pulsar2_ops.has_out_of_band_npu_data(compiled)


def test_bf16_checkpoint_compiles_to_a_real_axmodel(tmp_path):
    """Confirmed real, previously-broken case: a bare graph-level Cast from
    BFLOAT16 to FLOAT32 crashes a real pulsar2 build outright during its
    own frontend constant-folding pass (confirmed down to a standalone
    4-element tensor, independent of model size) -- see
    onnxsim/hf_reconstruct.py's module docstring. reconstruct_hf_graph()
    now upcasts BF16 weights to FLOAT32 in the stored initializer bytes
    instead of via a Cast node, specifically so this compiles. Found via a
    real HuggingFaceTB/SmolLM2-135M (135M params, 30 layers, real BF16
    checkpoint) build, reproduced here with a small synthetic BF16
    checkpoint so it doesn't need a real download."""
    hf_dir = tmp_path / "tiny_llama_bf16"
    hf_dir.mkdir()
    _write_tiny_llama_checkpoint(str(hf_dir), bf16=True)

    model = onnxsim.reconstruct_hf_graph(str(hf_dir), batch_size=1, seq_len=8)
    onnx.checker.check_model(model)
    # No initializer should be BFLOAT16 -- every weight is upcast to
    # FLOAT32 in the stored bytes, not via a graph-level Cast.
    assert not any(
        init.data_type == onnx.TensorProto.BFLOAT16 for init in model.graph.initializer
    )

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    result = pulsar2_docker.build_from_hf_checkpoint(
        str(hf_dir), str(work_dir), "output"
    )

    assert result.success, result.error
    assert result.axmodel_path is not None
    assert os.path.exists(result.axmodel_path)


def test_compiled_axmodel_runs_on_real_device_when_available(tmp_path):
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device connected")

    hf_dir = tmp_path / "tiny_llama"
    hf_dir.mkdir()
    _write_tiny_llama_checkpoint(str(hf_dir))
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    result = pulsar2_docker.build_from_hf_checkpoint(
        str(hf_dir), str(work_dir), "output"
    )
    assert result.success, result.error

    stats = pulsar2_docker.run_on_device(result.axmodel_path)
    assert stats["error"] is None
    assert stats["avg_ms"] is not None
    assert stats["avg_ms"] > 0
