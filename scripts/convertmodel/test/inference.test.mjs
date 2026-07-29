// Inference smoke test for the model converter.
//
// Runs the fixture model through onnxruntime-web for several iterations and
// checks that (a) every iteration produces identical output (execution is
// deterministic) and (b) the output matches the reference baked into io.json.
// It drives the same inference_core.mjs the browser converter page uses, so a
// green run here means the onnxruntime-web execution path works end to end.
//
// Usage:
//   node test/inference.test.mjs            # defaults: wasm EP, 5 iterations
//   ORT_EP=webgpu node test/inference.test.mjs
//   ORT_ITERS=20 node test/inference.test.mjs
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
const TOL = 1e-4;

// Load onnxruntime-web's wasm artifacts from the local install instead of a CDN
// so the test runs offline. Single-threaded, no worker proxy: the simplest
// configuration that works headless under Node.
ort.env.wasm.wasmPaths = DIST;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

async function main() {
  const model = new Uint8Array(readFileSync(join(HERE, "model.onnx")));
  const io = JSON.parse(readFileSync(join(HERE, "io.json"), "utf8"));

  console.log(`onnxruntime-web ${ort.env.versions.web ?? "?"}, EP=${EP}, iterations=${ITERS}`);

  const input = new ort.Tensor("float32", Float32Array.from(io.input.data), io.input.dims);
  const reference = Float32Array.from(io.output.data);

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

  console.log(
    `Y=[${Array.from(res.output).map((v) => v.toFixed(4)).join(", ")}]`,
  );
  console.log(
    `PASS: ${res.iterations} iterations on '${res.ep}', deterministic, ` +
      `within ${TOL} of reference (avg ${res.avgMs.toFixed(2)} ms/iter)`,
  );
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
