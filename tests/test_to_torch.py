"""onnxsim.to_torch: ONNX decoder LLM -> torch.export-able nn.Module.

The fixture is a one-layer decoder shaped like a HuggingFace optimum export
(``onnx-community/*`` models): token embedding, RMSNorm, RoPE computed from
``position_ids``, grouped-query attention with the ``Unsqueeze -> Expand -> Reshape``
repeat_kv spelling, ``past_key_values.*`` inputs concatenated into ``present.*``
outputs, and an additive causal mask built from ``Trilu``/``Shape`` of the past --
all with dynamic batch and sequence sizes, so every ``Reshape``/``Expand`` target is
computed from ``Shape`` at run time -- including HF's
``Gather(Shape(x), <scalar>) -> Unsqueeze -> Concat`` spelling, whose 0-d result is a
``torch.SymInt`` (not an ``int``) under ``torch.export``.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim.to_torch import onnx_state_dict, onnx_to_torch

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

VOCAB, HIDDEN, HQ, HK, HEAD = 16, 8, 2, 1, 4


def _decoder_model():
    body = f"""
    g (int64[B, S] input_ids, int64[B, T] attention_mask, int64[B, S] position_ids,
       float[B, {HK}, P, {HEAD}] past_key_values_0_key,
       float[B, {HK}, P, {HEAD}] past_key_values_0_value)
      => (float[B, S, {VOCAB}] logits, float[B, {HK}, T, {HEAD}] present_0_key,
          float[B, {HK}, T, {HEAD}] present_0_value)
    <float two = {{2.0}}, float eps = {{1e-6}}, float qk_scale = {{0.5}},
     float neg = {{-10000.0}}, float one_f = {{1.0}},
     int64[1] axes_last = {{-1}}, int64[1] ax0 = {{0}}, int64[1] ax1 = {{1}},
     int64[1] ax2 = {{2}}, int64[1] s0 = {{0}}, int64[1] s2 = {{2}},
     int64[1] big = {{9223372036854775807}}, int64[1] hq = {{{HQ}}},
     int64[1] hk = {{{HK}}}, int64[1] head = {{{HEAD}}}, int64[1] rep = {{{HQ // HK}}},
     int64[1] hidden = {{{HIDDEN}}}, int64[1] half = {{{HEAD // 2}}}, int64 one = {{1}}, int64 zero = {{0}}>
    {{
      x = Gather(embed, input_ids)
      xsq = Pow(x, two)
      var = ReduceMean<keepdims = 1>(xsq, axes_last)
      ve = Add(var, eps)
      rstd = Sqrt(ve)
      xn = Div(x, rstd)
      h = Mul(ln_w, xn)

      hs = Shape(h)
      bs = Slice(hs, s0, s2)
      q_shape = Concat<axis = 0>(bs, hq, head)
      k_shape = Concat<axis = 0>(bs, hk, head)
      q0 = MatMul(h, wq)
      q1 = Reshape(q0, q_shape)
      q = Transpose<perm = [0, 2, 1, 3]>(q1)
      k0 = MatMul(h, wk)
      k1 = Reshape(k0, k_shape)
      k = Transpose<perm = [0, 2, 1, 3]>(k1)
      v0 = MatMul(h, wv)
      v1 = Reshape(v0, k_shape)
      v = Transpose<perm = [0, 2, 1, 3]>(v1)

      pos_f = Cast<to = 1>(position_ids)
      pos_u = Unsqueeze(pos_f, axes_last)
      freqs = Mul(pos_u, inv_freq)
      emb = Concat<axis = -1>(freqs, freqs)
      cos0 = Cos(emb)
      sin0 = Sin(emb)
      cos = Unsqueeze(cos0, ax1)
      sin = Unsqueeze(sin0, ax1)
      q_lo = Slice(q, s0, half, axes_last)
      q_hi = Slice(q, half, big, axes_last)
      q_hi_n = Neg(q_hi)
      q_rot = Concat<axis = -1>(q_hi_n, q_lo)
      q_c = Mul(q, cos)
      q_s = Mul(q_rot, sin)
      q_r = Add(q_c, q_s)
      k_lo = Slice(k, s0, half, axes_last)
      k_hi = Slice(k, half, big, axes_last)
      k_hi_n = Neg(k_hi)
      k_rot = Concat<axis = -1>(k_hi_n, k_lo)
      k_c = Mul(k, cos)
      k_s = Mul(k_rot, sin)
      k_r = Add(k_c, k_s)

      present_0_key = Concat<axis = 2>(past_key_values_0_key, k_r)
      present_0_value = Concat<axis = 2>(past_key_values_0_value, v)

      ks = Shape(present_0_key)
      kb = Slice(ks, s0, ax1)
      kt = Slice(ks, s2, big, ax0)
      exp_shape = Concat<axis = 0>(kb, hk, rep, kt)
      rep_shape = Concat<axis = 0>(kb, hq, kt)
      k_u = Unsqueeze(present_0_key, ax2)
      k_e = Expand(k_u, exp_shape)
      k_rep = Reshape(k_e, rep_shape)
      v_u = Unsqueeze(present_0_value, ax2)
      v_e = Expand(v_u, exp_shape)
      v_rep = Reshape(v_e, rep_shape)

      k_t = Transpose<perm = [0, 1, 3, 2]>(k_rep)
      qk = MatMul(q_r, k_t)
      scores = Mul(qk, qk_scale)

      qs = Shape(q_r)
      seq = Gather<axis = 0>(qs, s2)
      total = Gather<axis = 0>(ks, s2)
      mask_shape = Concat<axis = 0>(seq, total)
      ones = ConstantOfShape<value = float[1] {{1.0}}>(mask_shape)
      ps = Shape(past_key_values_0_key)
      past_len = Gather<axis = 0>(ps, ax2)
      diag0 = Add(past_len, one)
      diag = Squeeze(diag0, ax0)
      upper = Trilu<upper = 1>(ones, diag)
      mask = Mul(upper, neg)
      am_f = Cast<to = 1>(attention_mask)
      am_u0 = Unsqueeze(am_f, ax1)
      am_u = Unsqueeze(am_u0, ax1)
      am_inv = Sub(one_f, am_u)
      am_add = Mul(am_inv, neg)
      full_mask = Add(mask, am_add)
      scores_m = Add(scores, full_mask)
      probs = Softmax<axis = -1>(scores_m)
      ctx = MatMul(probs, v_rep)
      ctx_t = Transpose<perm = [0, 2, 1, 3]>(ctx)
      b_dim = Gather<axis = 0>(hs, zero)
      s_dim = Gather<axis = 0>(hs, one)
      b_u = Unsqueeze(b_dim, ax0)
      s_u = Unsqueeze(s_dim, ax0)
      out_shape = Concat<axis = 0>(b_u, s_u, hidden)
      ctx_r = Reshape(ctx_t, out_shape)
      o = MatMul(ctx_r, wo)
      res = Add(x, o)
      logits = MatMul(res, lm_head)
    }}
    """
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["": 18]> {body}')
    rng = np.random.default_rng(0)

    def w(name, *shape, scale=0.3):
        return numpy_helper.from_array(
            (rng.standard_normal(shape) * scale).astype(np.float32), name
        )

    inv_freq = 1.0 / (10000 ** (np.arange(0, HEAD, 2) / HEAD))
    model.graph.initializer.extend(
        [
            w("embed", VOCAB, HIDDEN, scale=1.0),
            numpy_helper.from_array(
                (1 + 0.1 * rng.standard_normal(HIDDEN)).astype(np.float32), "ln_w"
            ),
            w("wq", HIDDEN, HQ * HEAD),
            w("wk", HIDDEN, HK * HEAD),
            w("wv", HIDDEN, HK * HEAD),
            w("wo", HQ * HEAD, HIDDEN),
            w("lm_head", HIDDEN, VOCAB),
            numpy_helper.from_array(inv_freq.astype(np.float32), "inv_freq"),
        ]
    )
    onnx.checker.check_model(model)
    return model


def _feeds(batch, seq, past, head=HEAD):
    rng = np.random.default_rng(batch * 100 + seq * 10 + past)
    return {
        "input_ids": rng.integers(0, VOCAB, (batch, seq)).astype(np.int64),
        "attention_mask": np.ones((batch, past + seq), np.int64),
        "position_ids": np.tile(np.arange(past, past + seq), (batch, 1)).astype(
            np.int64
        ),
        "past_key_values_0_key": rng.standard_normal((batch, HK, past, head)).astype(
            np.float32
        ),
        "past_key_values_0_value": rng.standard_normal((batch, HK, past, head)).astype(
            np.float32
        ),
    }


def _ort(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def test_exact_mode_matches_onnxruntime_with_kv_cache():
    # Faithful interpretation of every op, KV cache included: validates op coverage
    # and the shape-value arithmetic independently of attention recognition.
    model = _decoder_model()
    names = [i.name for i in model.graph.input]
    mod = onnx_to_torch(
        model,
        inputs=names,
        outputs=[o.name for o in model.graph.output],
        strip_kv_cache=False,
        attention="exact",
    )
    feeds = _feeds(2, 3, 4)
    want = _ort(model, feeds)
    got = mod(*(torch.from_numpy(feeds[n]) for n in names))
    for g, w in zip(got, want):
        np.testing.assert_allclose(g.numpy(), w, rtol=1e-5, atol=1e-5)


def test_sdpa_mode_strips_cache_and_mask():
    # Default mode: attention recognized and emitted as causal SDPA; the attention_mask
    # and past_key_values inputs are gone. Equals the ONNX model run as a prefill (empty
    # past, all-ones mask).
    model = _decoder_model()
    mod = onnx_to_torch(model)
    assert len(mod._attn) == 1
    feeds = _feeds(2, 5, 0)
    (want,) = _ort(model, feeds)[:1]
    (got,) = mod(
        torch.from_numpy(feeds["input_ids"]), torch.from_numpy(feeds["position_ids"])
    )
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-5)


def test_sdpa_mode_torch_export_dynamic_batch_and_seq():
    # The point of carrying shapes as Python (Sym)ints: torch.export with dynamic batch
    # and sequence succeeds, the graph has SDPA and no softmax/mask construction, and
    # the exported program is right at a different size than it was traced with.
    from torch.export import Dim, export

    model = _decoder_model()
    mod = onnx_to_torch(model)
    b, s = Dim("batch", max=64), Dim("seq", max=4096)
    feeds = _feeds(2, 5, 0)
    ep = export(
        mod,
        (),
        {
            "input_ids": torch.from_numpy(feeds["input_ids"]),
            "position_ids": torch.from_numpy(feeds["position_ids"]),
        },
        dynamic_shapes={"input_ids": {0: b, 1: s}, "position_ids": {0: b, 1: s}},
    )
    targets = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
    assert any("scaled_dot_product_attention" in t for t in targets), targets
    assert not any("softmax" in t or "triu" in t for t in targets), targets
    feeds = _feeds(3, 7, 0)
    (want,) = _ort(model, feeds)[:1]
    (got,) = ep.module()(
        input_ids=torch.from_numpy(feeds["input_ids"]),
        position_ids=torch.from_numpy(feeds["position_ids"]),
    )
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-5)


def test_meta_build_then_load_state_dict():
    # AutoDeploy's factories build on the meta device and load weights afterwards.
    model = _decoder_model()
    mod = onnx_to_torch(model, device="meta")
    assert all(p.device.type == "meta" for p in mod.parameters())
    mod.load_state_dict(onnx_state_dict(mod, model), assign=True)
    feeds = _feeds(1, 4, 0)
    (want,) = _ort(model, feeds)[:1]
    (got,) = mod(
        torch.from_numpy(feeds["input_ids"]), torch.from_numpy(feeds["position_ids"])
    )
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-5)


def test_live_use_of_stripped_past_is_an_error():
    # In exact mode the causal mask stays live, and it reads Shape(past_key_values_0_key);
    # with the cache stripped that must fail loudly at conversion time instead of
    # silently computing a different mask.
    model = _decoder_model()
    with pytest.raises(ValueError, match="stripped KV-cache input"):
        onnx_to_torch(
            model,
            inputs=["input_ids", "attention_mask", "position_ids"],
            attention="exact",
        )


def test_rope_table_sliced_by_past_length_is_whole_table():
    # HF's older rotary_emb slices its cos table to [:past_len + seq_len] before
    # indexing it with position_ids. With the cache stripped the past length is
    # unknown, and bounding the table by the *traced* seq_len would break every
    # decode step (seq_len 1, large positions) -- so the slice must become the whole
    # table, and a lone new token at position 7 must read row 7.
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 18]>
        g (int64[B, S] position_ids, float[B, 1, P, 2] past_key_values_0_key)
          => (float[B, S, 2] out)
        <int64[1] s0 = {0}, int64[1] ax0 = {0}, int64[1] ax2 = {2}>
        {
          ps = Shape(past_key_values_0_key)
          past_len = Gather<axis = 0>(ps, ax2)
          ss = Shape(position_ids)
          seq = Gather<axis = 0>(ss, ax2)
          total = Add(past_len, seq)
          table = Slice(cos_table, s0, total, ax0)
          out = Gather<axis = 0>(table, position_ids)
        }
        """
    )
    table = np.arange(32, dtype=np.float32).reshape(16, 2)
    model.graph.initializer.append(numpy_helper.from_array(table, "cos_table"))
    mod = onnx_to_torch(model, inputs=["position_ids"], outputs=["out"])
    (got,) = mod(torch.tensor([[7]]))
    np.testing.assert_array_equal(got.numpy(), table[[[7]]])


def test_opset23_attention_op():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 23]>
        g (float[B, 2, S, 4] q, float[B, 1, S, 4] k, float[B, 1, S, 4] v) => (float[B, 2, S, 4] y)
        { y = Attention<is_causal = 1>(q, k, v) }
        """
    )
    rng = np.random.default_rng(1)
    feeds = {
        n: rng.standard_normal(s).astype(np.float32)
        for n, s in (("q", (2, 2, 5, 4)), ("k", (2, 1, 5, 4)), ("v", (2, 1, 5, 4)))
    }
    mod = onnx_to_torch(model, inputs=["q", "k", "v"], outputs=["y"])
    (got,) = mod(*(torch.from_numpy(feeds[n]) for n in "qkv"))
    try:
        (want,) = _ort(model, feeds)
    except Exception as e:  # older onnxruntime: no opset-23 Attention kernel
        pytest.skip(f"onnxruntime cannot run opset-23 Attention: {e}")
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-5)


def test_unsupported_op_is_named():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        g (float[2, 3] x) => (float[2, 3] y) { y = Hardmax(x) }
        """
    )
    mod = onnx_to_torch(model, inputs=["x"], outputs=["y"])
    with pytest.raises(NotImplementedError, match="Hardmax"):
        mod(torch.zeros(2, 3))


# onnxruntime's GroupQueryAttention kernel needs a head size that is a multiple of 8.
GHEAD = 8


def _genai_model():
    # ONNX Runtime GenAI builder spelling (e.g. HuggingFaceTB/SmolLM2-360M-Instruct's
    # ONNX export): contrib RotaryEmbedding / GroupQueryAttention /
    # (Skip)SimplifiedLayerNormalization, past KV fed straight into GQA, and GQA's
    # seqlens_k / total_sequence_length derived from attention_mask.
    body = f"""
    g (int64[B, S] input_ids, int64[B, T] attention_mask, int64[B, S] position_ids,
       float[B, {HK}, P, {GHEAD}] past_key_values_0_key,
       float[B, {HK}, P, {GHEAD}] past_key_values_0_value)
      => (float[B, S, {VOCAB}] logits, float[B, {HK}, T, {GHEAD}] present_0_key,
          float[B, {HK}, T, {GHEAD}] present_0_value)
    <int64[1] one = {{1}}, int64 one_s = {{1}}>
    {{
      x = Gather(embed, input_ids)
      h = SimplifiedLayerNormalization<epsilon = 1e-6>(x, ln_w)
      q0 = MatMul(h, wq)
      k0 = MatMul(h, wk)
      v = MatMul(h, wv)
      q = com.microsoft.RotaryEmbedding(q0, position_ids, cos_cache, sin_cache)
      k = com.microsoft.RotaryEmbedding(k0, position_ids, cos_cache, sin_cache)
      am_sum = ReduceSum(attention_mask, one)
      am_len = Sub(am_sum, one)
      seqlens_k = Cast<to = 6>(am_len)
      am_shape = Shape(attention_mask)
      total0 = Gather<axis = 0>(am_shape, one_s)
      total = Cast<to = 6>(total0)
      ctx, present_0_key, present_0_value = com.microsoft.GroupQueryAttention<
          num_heads = {HQ}, kv_num_heads = {HK}, scale = 0.5>(
          q, k, v, past_key_values_0_key, past_key_values_0_value, seqlens_k, total)
      o = MatMul(ctx, wo)
      res_n, mean_unused, inv_unused, res = com.microsoft.SkipSimplifiedLayerNormalization<
          epsilon = 1e-6>(x, o, ln2_w)
      logits = MatMul(res_n, lm_head)
    }}
    """
    model = parser.parse_model(
        f'<ir_version: 8, opset_import: ["": 18, "com.microsoft": 1]> {body}'
    )
    rng = np.random.default_rng(3)

    def w(name, *shape, scale=0.3):
        return numpy_helper.from_array(
            (rng.standard_normal(shape) * scale).astype(np.float32), name
        )

    pos = np.arange(64)[:, None] / (10000 ** (np.arange(0, GHEAD, 2) / GHEAD))[None]
    model.graph.initializer.extend(
        [
            w("embed", VOCAB, HIDDEN, scale=1.0),
            w("ln_w", HIDDEN, scale=0.1),
            w("ln2_w", HIDDEN, scale=0.1),
            w("wq", HIDDEN, HQ * GHEAD),
            w("wk", HIDDEN, HK * GHEAD),
            w("wv", HIDDEN, HK * GHEAD),
            w("wo", HQ * GHEAD, HIDDEN),
            w("lm_head", HIDDEN, VOCAB),
            numpy_helper.from_array(np.cos(pos).astype(np.float32), "cos_cache"),
            numpy_helper.from_array(np.sin(pos).astype(np.float32), "sin_cache"),
        ]
    )
    return model


@pytest.mark.parametrize("past", [0, 3])
def test_genai_contrib_ops_exact_mode(past):
    # GQA with a real past, RoPE at offset positions: exact interpretation vs ORT's own
    # contrib kernels.
    model = _genai_model()
    names = [i.name for i in model.graph.input]
    mod = onnx_to_torch(
        model,
        inputs=names,
        outputs=[o.name for o in model.graph.output],
        strip_kv_cache=False,
        attention="exact",
    )
    # onnxruntime's GQA: batch must be 1 when a multi-token input has a past.
    feeds = _feeds(1 if past else 2, 4, past, head=GHEAD)
    try:
        want = _ort(model, feeds)
    except Exception as e:  # onnxruntime build without these contrib kernels
        pytest.skip(f"onnxruntime cannot run the contrib ops: {e}")
    got = mod(*(torch.from_numpy(feeds[n]) for n in names))
    for g, w in zip(got, want):
        np.testing.assert_allclose(g.numpy(), w, rtol=1e-4, atol=1e-4)


def test_genai_contrib_ops_sdpa_mode_and_export():
    # Default mode: GQA's past / seqlens inputs are dropped (never fetched), and the
    # module exports with dynamic batch and sequence.
    from torch.export import Dim, export

    model = _genai_model()
    mod = onnx_to_torch(model)
    feeds = _feeds(2, 5, 0, head=GHEAD)
    try:
        (want,) = _ort(model, feeds)[:1]
    except Exception as e:
        pytest.skip(f"onnxruntime cannot run the contrib ops: {e}")
    b, s = Dim("batch", max=64), Dim("seq", max=4096)
    ep = export(
        mod,
        (),
        {
            "input_ids": torch.from_numpy(feeds["input_ids"]),
            "position_ids": torch.from_numpy(feeds["position_ids"]),
        },
        dynamic_shapes={"input_ids": {0: b, 1: s}, "position_ids": {0: b, 1: s}},
    )
    (got,) = ep.module()(
        input_ids=torch.from_numpy(feeds["input_ids"]),
        position_ids=torch.from_numpy(feeds["position_ids"]),
    )
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("zero_points", [False, True])
def test_matmulnbits_dequantized_matches_onnxruntime(zero_points):
    # Weight-only int4 (onnx-community q4f16 / GenAI int4 exports): dequantized once at
    # conversion to a dense [K, N] weight. K=40 with block 16 also exercises the padded
    # last block; the bias input is exercised too.
    k, n, block = 40, 8, 16
    k_blocks = -(-k // block)
    rng = np.random.default_rng(7)
    packed = rng.integers(0, 256, (n, k_blocks, block // 2), dtype=np.uint8)
    scales = (rng.random(n * k_blocks) * 0.1 + 0.01).astype(np.float32)
    zp = rng.integers(0, 256, (n * ((k_blocks + 1) // 2),), dtype=np.uint8)
    bias = rng.standard_normal(n).astype(np.float32)
    zp_in = "zp" if zero_points else '""'
    model = parser.parse_model(
        f"""
        <ir_version: 8, opset_import: ["": 18, "com.microsoft": 1]>
        g (float[B, S, {k}] a) => (float[B, S, {n}] y) {{
          y = com.microsoft.MatMulNBits<K = {k}, N = {n}, bits = 4, block_size = {block}>(
              a, w, scales, {zp_in}, "", bias)
        }}
        """
    )
    inits = [
        numpy_helper.from_array(packed, "w"),
        numpy_helper.from_array(scales, "scales"),
        numpy_helper.from_array(bias, "bias"),
    ]
    if zero_points:
        inits.append(numpy_helper.from_array(zp, "zp"))
    model.graph.initializer.extend(inits)
    x = rng.standard_normal((2, 3, k)).astype(np.float32)
    try:
        (want,) = _ort(model, {"a": x})
    except Exception as e:
        pytest.skip(f"onnxruntime cannot run MatMulNBits: {e}")
    mod = onnx_to_torch(model, inputs=["a"], outputs=["y"])
    assert not any(t.dtype == torch.uint8 for t in mod.state_dict().values())
    (got,) = mod(torch.from_numpy(x))
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-4)
    meta = onnx_to_torch(model, inputs=["a"], outputs=["y"], device="meta")
    meta.load_state_dict(onnx_state_dict(meta.param_names, model), assign=True)
    np.testing.assert_allclose(
        meta(torch.from_numpy(x))[0].numpy(), want, rtol=1e-4, atol=1e-4
    )
