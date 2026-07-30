// Inference smoke test for the model converter.
//
// Runs the fixture model through onnxruntime-web for several iterations and
// checks that (a) every iteration produces identical output (execution is
// deterministic) and (b) the output matches the reference baked into io.json.
// It drives the same inference_core.mjs the browser converter page uses, so a
// green run here means the onnxruntime-web execution path works end to end.
//
// Usage:
//   node test/inference.test.mjs            # defaults: wasm EP, 5 iterations, batch 1
//   ORT_EP=webgpu node test/inference.test.mjs
//   ORT_ITERS=20 node test/inference.test.mjs
//   ORT_BATCH=64 node test/inference.test.mjs   # feed 64 input rows
//
// The fixture's batch (first) input dimension is symbolic, so ORT_BATCH picks
// how many rows to feed. Each row runs independently through MatMul/Add/Relu, so
// we replicate the single reference row baked into io.json ORT_BATCH times and
// every output row must still match it.
//
// WebGPU is only available where a GPU + WebGPU runtime exists (e.g. a browser);
// in a plain CI/Node container use wasm.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as ort from "onnxruntime-web";
import { runInference } from "../inference_core.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = join(HERE, "..", "node_modules", "onnxruntime-web", "dist") + "/";

const EP = process.env.ORT_EP || "wasm";
const ITERS = Number(process.env.ORT_ITERS || "5");
const BATCH = Math.max(1, Math.floor(Number(process.env.ORT_BATCH || "1")) || 1);
const TOL = 1e-4;

// Repeat `row` `n` times back-to-back into a flat array (row-major tiling of the
// batch dimension).
function tileRows(row, n) {
  const out = new Array(row.length * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < row.length; j++) out[i * row.length + j] = row[j];
  }
  return out;
}

// Load onnxruntime-web's wasm artifacts from the local install instead of a CDN
// so the test runs offline. Single-threaded, no worker proxy: the simplest
// configuration that works headless under Node.
ort.env.wasm.wasmPaths = DIST;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

async function main() {
  const model = new Uint8Array(readFileSync(join(HERE, "model.onnx")));
  const io = JSON.parse(readFileSync(join(HERE, "io.json"), "utf8"));

  console.log(
    `onnxruntime-web ${ort.env.versions.web ?? "?"}, EP=${EP}, ` +
      `iterations=${ITERS}, batch=${BATCH}`,
  );

  // io.json holds one reference row; replicate it BATCH times to size the input.
  // dims[0] is the (symbolic) batch dimension, so only it scales with BATCH.
  const inputDims = [BATCH, ...io.input.dims.slice(1)];
  const input = new ort.Tensor(
    "float32",
    Float32Array.from(tileRows(io.input.data, BATCH)),
    inputDims,
  );
  const reference = Float32Array.from(tileRows(io.output.data, BATCH));

  const res = await runInference(ort, {
    model,
    inputName: io.input.name,
    input,
    outputName: io.output.name,
    providers: [EP],
    iterations: ITERS,
    reference,
    tolerance: TOL,
    onLog: (m) => console.log("  " + m),
  });

  // Every row is identical, so show just the first one (row width = ref length).
  const rowLen = io.output.data.length;
  const firstRow = Array.from(res.output).slice(0, rowLen);
  console.log(
    `Y[0]=[${firstRow.map((v) => v.toFixed(4)).join(", ")}]` +
      (BATCH > 1 ? ` (x${BATCH} rows)` : ""),
  );
  console.log(
    `PASS: ${res.iterations} iterations on '${res.ep}', batch ${BATCH}, ` +
      `deterministic, within ${TOL} of reference (avg ${res.avgMs.toFixed(2)} ms/iter)`,
  );
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
