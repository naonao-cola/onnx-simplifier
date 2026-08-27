// End-to-end test for the onnx_deploy_wasm module: loads the compiled
// ../build/onnx_deploy_wasm.js, wires up the onnxruntime-web-backed runtime
// (ort_web_runtime.mjs), runs generate() against the same hand-built toy
// seq2seq export ../../scripts/make_toy_seq2seq.py produces (see that
// script's docstring for why its output is fully hand-computable), and
// asserts the exact same token sequence the native CLI/Python extension
// produce for the same inputs (see ../../README.md's "Verifying the flow").
//
// Usage: node run_test.mjs <toy_model_dir>
//   (run ../../scripts/make_toy_seq2seq.py -o <toy_model_dir> first)

import { readFileSync } from "node:fs";
import { strict as assert } from "node:assert";
import path from "node:path";

import { installRuntime } from "./ort_web_runtime.mjs";
import createOnnxDeployWasmModule from "../build/onnx_deploy_wasm.js";

const modelDir = process.argv[2];
if (!modelDir) {
  console.error("usage: node run_test.mjs <toy_model_dir>");
  process.exit(1);
}

const encoderBytes = new Uint8Array(readFileSync(path.join(modelDir, "encoder_model.onnx")));
const decoderBytes = new Uint8Array(readFileSync(path.join(modelDir, "decoder_model.onnx")));
const decoderPastBytes = new Uint8Array(readFileSync(path.join(modelDir, "decoder_with_past_model.onnx")));

const moduleConfig = installRuntime({});
const Module = await createOnnxDeployWasmModule(moduleConfig);

async function run(maxNewTokens, eosTokenId) {
  const result = await Module.generate(encoderBytes, decoderBytes, decoderPastBytes, [3, 4], maxNewTokens, eosTokenId, 0);
  return Array.from(result, (v) => Number(v));
}

const full = await run(8, -1);
console.log("generate(max_new_tokens=8, eos=-1):", full);
assert.deepEqual(full, [0, 1, 2, 3, 4, 5, 6, 0], "full 8-token sequence mismatch");

const early = await run(20, 6);
console.log("generate(max_new_tokens=20, eos_token_id=6):", early);
assert.deepEqual(early, [0, 1, 2, 3, 4, 5, 6], "eos_token_id early-stop mismatch");

console.log("OK: onnx_deploy_wasm matches the native pipeline's expected output.");
