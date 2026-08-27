#!/usr/bin/env python3
"""Builds a tiny, hand-crafted seq2seq export matching optimum-onnx's
no_post_process=True file shape (encoder_model.onnx / decoder_model.onnx /
decoder_with_past_model.onnx) and I/O naming convention, WITHOUT needing
torch/transformers/optimum -- just the `onnx` package, mirroring
tools/onnx-finetune/scripts/make_toy_model.py's role for that tool.

This is not a real language model: the "logits" are a deterministic,
hand-computable function of the encoder input and step count, specifically
chosen so a correct KvCachePipeline run has one and only one possible output
sequence -- see compute_expected_ids() below and README.md's "Verifying the
flow" section for the derivation. It exists to give tools/onnx-deploy an
end-to-end regression test with no heavy ML-framework dependency.

Model (single "layer", hidden dim 1):
  encoder_hidden_states = Cast(encoder input_ids, float), one scalar per
    encoder position, unsqueezed to [1, enc_seq, 1].
  decoder_model.onnx (step 0, no past):
    self_state  = Cast(decoder input_ids[0], float)      -- seeds the
      self-attention cache from the decoder start token.
    cross_state = ReduceSum(encoder_hidden_states)        -- the "encoder
      cache": computed once here, reused unchanged every step after (this
      graph is the only one that outputs present.0.encoder.*).
    present.0.decoder.{key,value} = self_state
    present.0.encoder.{key,value} = cross_state
    logits = OneHot(Mod(self_state + cross_state, vocab_size), vocab_size)
  decoder_with_past_model.onnx (every step after):
    new_self = past_key_values.0.decoder.key + 1          -- deliberately
      ignores the fed input_ids and just increments, so the generated
      sequence directly reflects whether the self-attention cache is really
      being threaded from one Run() call to the next (a broken pipeline
      that always re-feeds a zeroed/empty cache would produce a different,
      non-incrementing sequence instead).
    cross = past_key_values.0.encoder.key                 -- passed through
      from the cache untouched; this graph does NOT re-output
      present.0.encoder.* (matching real T5-style cross-attention caches),
      which is exactly the case that requires KvCachePipeline to keep an
      unrefreshed cache entry alive across steps instead of dropping it
      once it isn't part of one call's outputs.
    present.0.decoder.{key,value} = new_self
    logits = OneHot(Mod(new_self + cross, vocab_size), vocab_size)

Usage:
    python3 make_toy_seq2seq.py -o toy_seq2seq --vocab-size 7 \\
        --encoder-ids 3,4 --decoder-start-token-id 0
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, checker, helper


def _onehot_logits_subgraph(builder_prefix, state_name, vocab_size):
    """Nodes computing logits = OneHot(Mod(Cast(state, int64), vocab_size), vocab_size),
    where `state_name` is a float [1, 1, 1] tensor. Returns (nodes, logits_name)."""
    state_i64 = f"{builder_prefix}_state_i64"
    index3 = f"{builder_prefix}_index3"
    index2 = f"{builder_prefix}_index2"
    logits = "logits"
    nodes = [
        helper.make_node("Cast", [state_name], [state_i64], to=TensorProto.INT64),
        helper.make_node("Mod", [state_i64, f"{builder_prefix}_vocab_size"], [index3]),
        helper.make_node("Squeeze", [index3, f"{builder_prefix}_axis2"], [index2]),
        helper.make_node(
            "OneHot", [index2, f"{builder_prefix}_depth", f"{builder_prefix}_onehot_values"], [logits], axis=-1
        ),
    ]
    initializers = [
        helper.make_tensor(f"{builder_prefix}_vocab_size", TensorProto.INT64, [1, 1, 1], [vocab_size] * 1),
        helper.make_tensor(f"{builder_prefix}_axis2", TensorProto.INT64, [1], [2]),
        helper.make_tensor(f"{builder_prefix}_depth", TensorProto.INT64, [], [vocab_size]),
        helper.make_tensor(f"{builder_prefix}_onehot_values", TensorProto.FLOAT, [2], [0.0, 1.0]),
    ]
    return nodes, initializers, logits


def make_encoder_model(enc_seq_len):
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, enc_seq_len])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, enc_seq_len])
    encoder_hidden_states = helper.make_tensor_value_info(
        "encoder_hidden_states", TensorProto.FLOAT, [1, enc_seq_len, 1]
    )
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axis_neg1"], ["encoder_hidden_states"]),
    ]
    initializers = [helper.make_tensor("axis_neg1", TensorProto.INT64, [1], [-1])]
    graph = helper.make_graph(
        nodes, "toy_encoder", [input_ids, attention_mask], [encoder_hidden_states], initializer=initializers
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def make_decoder_model(enc_seq_len, vocab_size):
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])
    encoder_attention_mask = helper.make_tensor_value_info(
        "encoder_attention_mask", TensorProto.INT64, [1, enc_seq_len]
    )
    encoder_hidden_states = helper.make_tensor_value_info(
        "encoder_hidden_states", TensorProto.FLOAT, [1, enc_seq_len, 1]
    )

    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axis_neg1"], ["self_state"]),
        helper.make_node("ReduceSum", ["encoder_hidden_states", "axis1"], ["cross_state"], keepdims=1),
        helper.make_node("Identity", ["self_state"], ["present.0.decoder.key"]),
        helper.make_node("Identity", ["self_state"], ["present.0.decoder.value"]),
        helper.make_node("Identity", ["cross_state"], ["present.0.encoder.key"]),
        helper.make_node("Identity", ["cross_state"], ["present.0.encoder.value"]),
        helper.make_node("Add", ["self_state", "cross_state"], ["accumulated"]),
    ]
    initializers = [
        helper.make_tensor("axis_neg1", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
    ]
    onehot_nodes, onehot_inits, logits_name = _onehot_logits_subgraph("d0", "accumulated", vocab_size)
    nodes += onehot_nodes
    initializers += onehot_inits
    assert logits_name == "logits"

    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab_size]),
        helper.make_tensor_value_info("present.0.decoder.key", TensorProto.FLOAT, [1, 1, 1]),
        helper.make_tensor_value_info("present.0.decoder.value", TensorProto.FLOAT, [1, 1, 1]),
        helper.make_tensor_value_info("present.0.encoder.key", TensorProto.FLOAT, [1, 1, 1]),
        helper.make_tensor_value_info("present.0.encoder.value", TensorProto.FLOAT, [1, 1, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "toy_decoder",
        [input_ids, encoder_attention_mask, encoder_hidden_states],
        outputs,
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def make_decoder_with_past_model(enc_seq_len, vocab_size):
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])
    encoder_attention_mask = helper.make_tensor_value_info(
        "encoder_attention_mask", TensorProto.INT64, [1, enc_seq_len]
    )
    past_decoder_key = helper.make_tensor_value_info("past_key_values.0.decoder.key", TensorProto.FLOAT, [1, 1, 1])
    past_decoder_value = helper.make_tensor_value_info(
        "past_key_values.0.decoder.value", TensorProto.FLOAT, [1, 1, 1]
    )
    past_encoder_key = helper.make_tensor_value_info("past_key_values.0.encoder.key", TensorProto.FLOAT, [1, 1, 1])
    past_encoder_value = helper.make_tensor_value_info(
        "past_key_values.0.encoder.value", TensorProto.FLOAT, [1, 1, 1]
    )

    nodes = [
        # Deliberately ignores input_ids for the increment -- see module
        # docstring: this makes the generated sequence a direct witness of
        # whether the self-attention cache is threaded correctly.
        helper.make_node("Add", ["past_key_values.0.decoder.key", "one_const"], ["new_self"]),
        helper.make_node("Identity", ["new_self"], ["present.0.decoder.key"]),
        helper.make_node("Identity", ["new_self"], ["present.0.decoder.value"]),
        # NOT re-output as present.0.encoder.* -- exercises KvCachePipeline
        # keeping this cache entry alive across steps on its own.
        helper.make_node("Identity", ["past_key_values.0.encoder.key"], ["cross"]),
        helper.make_node("Add", ["new_self", "cross"], ["accumulated"]),
    ]
    initializers = [helper.make_tensor("one_const", TensorProto.FLOAT, [1, 1, 1], [1.0])]
    onehot_nodes, onehot_inits, logits_name = _onehot_logits_subgraph("dp0", "accumulated", vocab_size)
    nodes += onehot_nodes
    initializers += onehot_inits
    assert logits_name == "logits"

    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, vocab_size]),
        helper.make_tensor_value_info("present.0.decoder.key", TensorProto.FLOAT, [1, 1, 1]),
        helper.make_tensor_value_info("present.0.decoder.value", TensorProto.FLOAT, [1, 1, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "toy_decoder_with_past",
        [input_ids, encoder_attention_mask, past_decoder_key, past_decoder_value, past_encoder_key, past_encoder_value],
        outputs,
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def compute_expected_ids(encoder_ids, decoder_start_token_id, vocab_size, max_new_tokens, eos_token_id=None):
    """Reference implementation of the toy model's math, independent of ONNX
    Runtime, for asserting against KvCachePipeline's actual output."""
    cross = float(sum(encoder_ids))
    self_state = float(decoder_start_token_id)
    generated = []
    for step in range(max_new_tokens):
        if step > 0:
            self_state = self_state + 1.0
        accumulated = self_state + cross
        token = int(accumulated) % vocab_size
        generated.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=7)
    parser.add_argument("--encoder-ids", default="3,4", help="comma-separated encoder input token ids")
    parser.add_argument("--decoder-start-token-id", type=int, default=0)
    args = parser.parse_args()

    encoder_ids = [int(x) for x in args.encoder_ids.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    models = {
        "encoder_model.onnx": make_encoder_model(len(encoder_ids)),
        "decoder_model.onnx": make_decoder_model(len(encoder_ids), args.vocab_size),
        "decoder_with_past_model.onnx": make_decoder_with_past_model(len(encoder_ids), args.vocab_size),
    }
    for name, model in models.items():
        checker.check_model(model)
        onnx.save(model, os.path.join(args.output_dir, name))
        print(f"wrote {name}")

    print(f"encoder_ids={encoder_ids} decoder_start_token_id={args.decoder_start_token_id} "
          f"vocab_size={args.vocab_size}")


if __name__ == "__main__":
    main()
