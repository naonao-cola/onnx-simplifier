# Integration/regression test: onnxsim.export_causal_lm_static_cache against
# a real Hugging Face `transformers` decoder-only causal LM, run through the
# actual deployment flow (prefill once, decode-with-fixed-buffer-KV-cache
# repeatedly via onnxruntime), compared token-for-token against the standard
# growing-cache export driven the same way.
#
# See onnxsim/transformers_export.py's own module docstring for
# export_causal_lm_static_cache's full rationale: optimum-onnx's own
# decoder-with-past export grows its past_key_values/present tensors by one
# position every decode step (Concat), so every step reallocates and copies
# the *entire* cache seen so far. This function exports a fixed-size
# ("static") KV-cache pair instead -- prefill.onnx (variable prompt length,
# empty cache) and decode.onnx (one new token, cache partially filled) --
# using transformers.StaticCache directly (not optimum's own OnnxConfig
# machinery), which already gets the causal-mask math right for a
# fixed-size buffer (confirmed necessary the hard way: a post-hoc
# Concat->TensorScatter rewrite of a growing-cache export hits a broadcast
# error, because HF's masking_utils derives the mask's size from the cache's
# *real* valid length, which stops matching the *tensor's* length once that
# tensor is a fixed-size buffer).
#
# `torch` and `transformers` are NOT normal test dependencies (heavy, and
# optional -- see onnxsim[transformers]); `onnxruntime` is needed here too,
# to actually drive the exported graphs (onnxsim's own C++ core never links
# against it for building/simplifying, only this test's *verification*
# does). This test skips unless all three are already importable, and skips
# (rather than fails) on a network error downloading the tiny model,
# matching tests/test_optimum_export_deploy.py's own convention.
#
# To run it locally::
#
#     pip install torch transformers onnxruntime onnxscript
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_static_kv_cache_export.py -v

import numpy as np
import pytest

import onnxsim

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
onnxruntime = pytest.importorskip("onnxruntime")

_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
_MAX_CACHE_LEN = 32
_PROMPT_LEN = 7
_N_DECODE_STEPS = 8


@pytest.fixture(scope="module")
def hf_model_config():
    try:
        config = transformers.AutoConfig.from_pretrained(_MODEL_ID)
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(
            f"Could not fetch config for {_MODEL_ID} from Hugging Face Hub: {e}"
        )
    return config


def _num_layers_heads_head_dim(config):
    text_config = config.get_text_config(decoder=True)
    num_layers = text_config.num_hidden_layers
    num_kv_heads = (
        getattr(text_config, "num_key_value_heads", None)
        or text_config.num_attention_heads
    )
    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads
    )
    return num_layers, num_kv_heads, head_dim


def _run_growing_cache_baseline(model_dir, config, prompt_ids):
    num_layers, num_kv_heads, head_dim = _num_layers_heads_head_dim(config)
    sess = onnxruntime.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    past = {
        f"past_key_values.{i}.{kind}": np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        for i in range(num_layers)
        for kind in ("key", "value")
    }
    generated = []
    input_ids = prompt_ids
    total_len = 0
    for _ in range(1 + _N_DECODE_STEPS):
        seq_len = input_ids.shape[1]
        position_ids = np.arange(total_len, total_len + seq_len)[None, :].astype(
            np.int64
        )
        attention_mask = np.ones((1, total_len + seq_len), dtype=np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past,
        }
        outputs = sess.run(None, feeds)
        out = dict(zip([o.name for o in sess.get_outputs()], outputs))
        next_token = int(np.argmax(out["logits"][0, -1]))
        generated.append(next_token)
        total_len += seq_len
        past = {
            f"past_key_values.{i}.{kind}": out[f"present.{i}.{kind}"]
            for i in range(num_layers)
            for kind in ("key", "value")
        }
        input_ids = np.array([[next_token]], dtype=np.int64)
    return generated


def _run_static_cache_export(export_dir, config, prompt_ids):
    num_layers, num_kv_heads, head_dim = _num_layers_heads_head_dim(config)
    prefill_sess = onnxruntime.InferenceSession(
        str(export_dir / "prefill.onnx"), providers=["CPUExecutionProvider"]
    )
    decode_sess = onnxruntime.InferenceSession(
        str(export_dir / "decode.onnx"), providers=["CPUExecutionProvider"]
    )

    past_kv = {}
    for i in range(num_layers):
        past_kv[f"past_key.{i}"] = np.zeros(
            (1, num_kv_heads, _MAX_CACHE_LEN, head_dim), dtype=np.float32
        )
        past_kv[f"past_value.{i}"] = np.zeros(
            (1, num_kv_heads, _MAX_CACHE_LEN, head_dim), dtype=np.float32
        )

    def run_step(sess, input_ids, cache_position, attention_mask, past_kv):
        feeds = {
            "input_ids": input_ids,
            "cache_position": cache_position,
            "attention_mask": attention_mask,
        }
        feeds.update(past_kv)
        outputs = sess.run(None, feeds)
        out = dict(zip([o.name for o in sess.get_outputs()], outputs))
        new_kv = {f"past_key.{i}": out[f"present_key.{i}"] for i in range(num_layers)}
        new_kv.update(
            {f"past_value.{i}": out[f"present_value.{i}"] for i in range(num_layers)}
        )
        return out["logits"], new_kv

    generated = []
    prompt_len = prompt_ids.shape[1]
    cache_position = np.arange(0, prompt_len).astype(np.int64)
    attention_mask = np.zeros((1, _MAX_CACHE_LEN), dtype=np.int64)
    attention_mask[:, :prompt_len] = 1
    logits, past_kv = run_step(
        prefill_sess, prompt_ids, cache_position, attention_mask, past_kv
    )
    next_token = int(np.argmax(logits[0, -1]))
    generated.append(next_token)
    write_pos = prompt_len

    for _ in range(_N_DECODE_STEPS):
        input_ids = np.array([[next_token]], dtype=np.int64)
        cache_position = np.array([write_pos], dtype=np.int64)
        attention_mask = np.zeros((1, _MAX_CACHE_LEN), dtype=np.int64)
        attention_mask[:, : write_pos + 1] = 1
        logits, past_kv = run_step(
            decode_sess, input_ids, cache_position, attention_mask, past_kv
        )
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)
        write_pos += 1

    return generated


def test_static_cache_export_matches_growing_cache_baseline(hf_model_config, tmp_path):
    optimum_exporters_onnx = pytest.importorskip("optimum.exporters.onnx")
    baseline_dir = tmp_path / "baseline"
    try:
        optimum_exporters_onnx.main_export(
            _MODEL_ID,
            output=str(baseline_dir),
            task="text-generation-with-past",
            no_post_process=True,
        )
    except Exception as e:
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    export_dir = tmp_path / "static_export"
    results = onnxsim.export_causal_lm_static_cache(
        _MODEL_ID,
        str(export_dir),
        max_cache_len=_MAX_CACHE_LEN,
        check_n=1,
    )
    assert set(results.keys()) == {"prefill.onnx", "decode.onnx"}
    assert all(results.values()), f"onnxsim numerical check failed: {results}"

    rng = np.random.RandomState(0)
    prompt_ids = rng.randint(
        0, min(900, hf_model_config.vocab_size), size=(1, _PROMPT_LEN)
    ).astype(np.int64)

    baseline_tokens = _run_growing_cache_baseline(
        baseline_dir, hf_model_config, prompt_ids
    )
    static_tokens = _run_static_cache_export(export_dir, hf_model_config, prompt_ids)

    assert static_tokens == baseline_tokens, (
        f"static-cache export diverged from growing-cache baseline: "
        f"{static_tokens} != {baseline_tokens}"
    )
