// Before/after comparison test for the model converter's "Run inference"
// panel.
//
// The panel's "compare (before vs after conversion)" mode feeds one shared input
// to two models — the original upload and the simplify/optimize result — and
// reports how far their outputs diverge (max |Δ|) plus the speed difference. It
// drives compareInference()/compareOutputs() in inference_core.mjs, the same
// code the browser page calls, so a green run here means that comparison path
// works end to end.
//
// Two checks:
//   1. compareInference() with the fixture model as BOTH sides (a conversion is
//      free to be a no-op) must report the outputs as identical (max |Δ| == 0),
//      dims matching, and equivalent within tolerance — the "conversion
//      preserved the numerics" verdict.
//   2. compareOutputs() alone (pure, no onnxruntime) reports the divergent and
//      not-comparable branches the way the panel renders them.
//
// Usage:
//   node test/compare.test.mjs            # wasm EP, 5 iterations
//   ORT_EP=webgpu node test/compare.test.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";
import * as ort from "onnxruntime-web";
import { compareInference, compareOutputs } from "../inference_core.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = join(HERE, "..", "node_modules", "onnxruntime-web", "dist") + "/";

const EP = process.env.ORT_EP || "wasm";
const ITERS = Number(process.env.ORT_ITERS || "5");
const TOL = 1e-3;

// Same offline, single-threaded onnxruntime-web config as inference.test.mjs.
ort.env.wasm.wasmPaths = DIST;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

async function testEndToEnd() {
  const model = new Uint8Array(readFileSync(join(HERE, "model.onnx")));
  const io = JSON.parse(readFileSync(join(HERE, "io.json"), "utf8"));

  const input = new ort.Tensor(
    "float32",
    Float32Array.from(io.input.data),
    io.input.dims,
  );

  // Compare the fixture model against itself: a conversion that changed nothing
  // is the strongest "must be equivalent" case, and it exercises the full
  // two-run + diff path without needing a second fixture.
  const cmp = await compareInference(ort, {
    before: { model, label: "original" },
    after: { model, label: "converted" },
    inputName: io.input.name,
    outputName: io.output.name,
    input,
    providers: [EP],
    iterations: ITERS,
    tolerance: TOL,
    onLog: (m) => console.log("  " + m),
  });

  assert.equal(cmp.comparable, true, "outputs should be comparable");
  assert.equal(cmp.dimsMatch, true, "output dims should match");
  assert.equal(cmp.maxAbsDiff, 0, "identical models must diverge by exactly 0");
  assert.equal(cmp.equivalent, true, "identical models must be equivalent");
  assert.ok(cmp.speedup > 0, "speedup should be a positive ratio");
  // Both runs must also match the reference row baked into io.json.
  const ref = Float32Array.from(io.output.data);
  const firstAfter = Array.from(cmp.after.output);
  let maxRefDiff = 0;
  for (let i = 0; i < ref.length; i++) {
    maxRefDiff = Math.max(maxRefDiff, Math.abs(firstAfter[i] - ref[i]));
  }
  assert.ok(maxRefDiff <= 1e-4, `output within ${1e-4} of reference (got ${maxRefDiff})`);

  console.log(
    `PASS end-to-end: both models ran ${cmp.after.iterations} iterations on ` +
      `'${cmp.after.ep}', max |Δ| = ${cmp.maxAbsDiff}, ` +
      `speedup ${cmp.speedup.toFixed(2)}×`,
  );
}

function testCompareOutputs() {
  // Divergent outputs beyond tolerance -> comparable but not equivalent.
  const differ = compareOutputs(
    { output: new Float32Array([0, 0, 0]), dims: [1, 3], avgMs: 2 },
    { output: new Float32Array([0, 0.5, 0]), dims: [1, 3], avgMs: 1 },
    TOL,
  );
  assert.equal(differ.comparable, true);
  assert.equal(differ.dimsMatch, true);
  assert.ok(Math.abs(differ.maxAbsDiff - 0.5) < 1e-9);
  assert.equal(differ.equivalent, false, "0.5 diff exceeds tolerance");
  assert.ok(Math.abs(differ.speedup - 2) < 1e-9, "2ms -> 1ms is a 2x speedup");

  // Within tolerance -> equivalent.
  const same = compareOutputs(
    { output: new Float32Array([1, 2, 3]), dims: [1, 3], avgMs: 1 },
    { output: new Float32Array([1, 2, 3 + 1e-6]), dims: [1, 3], avgMs: 1 },
    TOL,
  );
  assert.equal(same.equivalent, true, "tiny diff is within tolerance");

  // Mismatched output lengths -> not comparable, maxAbsDiff null.
  const uncomparable = compareOutputs(
    { output: new Float32Array([0, 0]), dims: [1, 2], avgMs: 1 },
    { output: new Float32Array([0, 0, 0]), dims: [1, 3], avgMs: 1 },
    TOL,
  );
  assert.equal(uncomparable.comparable, false);
  assert.equal(uncomparable.maxAbsDiff, null);
  assert.equal(uncomparable.dimsMatch, false);
  assert.equal(uncomparable.equivalent, false);

  console.log("PASS compareOutputs: differ / equivalent / not-comparable branches");
}

async function main() {
  console.log(
    `onnxruntime-web ${ort.env.versions.web ?? "?"}, EP=${EP}, iterations=${ITERS}`,
  );
  testCompareOutputs();
  await testEndToEnd();
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
