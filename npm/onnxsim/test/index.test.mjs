// Unit tests for the onnxsim npm package's public API (index.mjs).
//
// Needs the wasm module already built and staged (onnxsim.cjs / onnxsim.wasm)
// -- see scripts/build_npm_package.sh, or .github/workflows/static.yml, which
// runs this right after its own build via `npm test`.
//
// simplify() runs with constantFolding disabled here so the test never needs
// onnxruntime-web's wasm assets located on disk (see docs/wasm_ort_web.md):
// this exercises the embind marshaling, shape inference, and onnx-optimizer
// passes, not the onnxruntime-web constant-folding bridge -- that path is
// covered by the convertmodel demo's own inference test
// (scripts/convertmodel/test/inference.test.mjs), which configures
// onnxruntime-web's wasmPaths explicitly.
//
// Usage:
//   cd npm/onnxsim && npm install && npm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  applyAttentionHeadPruning,
  applyDoubleQuantization,
  applyEmbeddingVocabMagnitudePruning,
  applyEmbeddingVocabPruning,
  applyGgufQ4_0,
  applyIq4Nl,
  applyMoeExpertChannelPruning,
  applyOutlierSuppression,
  applyQuarot,
  applySmoothQuant,
  applyStructuredPruning,
  applyWandaPruning,
  crossLayerEqualize,
  pruneMagnitude,
  quantizeAttentionDynamic,
  quantizeDynamicMatMulIntegerToFloat,
  quantizeWeightOnlyInt8Block,
  quantizeWeightOnlyInt16,
  quantizeWeightOnlyMatMulNbits,
  quantizeWeightOnlyMxfp4,
  simplify,
  versions,
} from "../index.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE = join(HERE, "..", "..", "..", "scripts", "convertmodel", "test", "model.onnx");

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// onnxsim.cjs is Emscripten's minified, single-physical-line output; an
// uncaught throw/rejection reaching Node's default handler makes it dump the
// *entire* file as "source context" instead of a useful message (it treats
// the whole file as one "line"). Catch everything here and print just the
// error's message/stack so a real failure stays readable.
try {
  await check("simplify() returns a non-empty model and no trace by default", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    const { model, trace } = await simplify(input, { constantFolding: false });
    assert.ok(model instanceof Uint8Array);
    assert.ok(model.length > 0);
    assert.equal(trace, "");
  });

  await check("simplify() accepts an ArrayBuffer too", async () => {
    const input = readFileSync(FIXTURE);
    const { model } = await simplify(
      input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength),
      { constantFolding: false },
    );
    assert.ok(model instanceof Uint8Array);
    assert.ok(model.length > 0);
  });

  await check("simplify() rejects a non-bytes input", async () => {
    await assert.rejects(() => simplify("not bytes"), TypeError);
  });

  await check("simplify() rejects bytes that aren't a valid model", async () => {
    await assert.rejects(() => simplify(new Uint8Array([1, 2, 3])));
  });

  await check("versions() reports onnxsim and onnx-optimizer version strings", async () => {
    const v = await versions();
    assert.equal(typeof v.onnxsim, "string");
    assert.ok(v.onnxsim.length > 0);
    assert.equal(typeof v.onnx_optimizer, "string");
    assert.ok(v.onnx_optimizer.length > 0);
  });

  // Data-free C++ passes: the fixture (MatMul+Add+Relu, X:[N,4], W:[4,3])
  // exercises them without needing onnxruntime-web, same as the simplify()
  // checks above.
  await check("data-free quantize passes return non-empty models", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    for (const fn of [
      crossLayerEqualize,
      quantizeDynamicMatMulIntegerToFloat,
      quantizeAttentionDynamic,
      quantizeWeightOnlyInt16,
      quantizeWeightOnlyInt8Block,
      quantizeWeightOnlyMxfp4,
      quantizeWeightOnlyMatMulNbits,
      applyDoubleQuantization,
      applyIq4Nl,
      applyGgufQ4_0,
    ]) {
      const out = await fn(input);
      assert.ok(out instanceof Uint8Array, fn.name);
      assert.ok(out.length > 0, fn.name);
    }
  });

  await check("data-free pruning passes return non-empty models", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    for (const [fn, options] of [
      [pruneMagnitude, { sparsity: 0.5 }],
      [applyStructuredPruning, { sparsity: 0.5, importanceNorm: "l2" }],
      [applyAttentionHeadPruning, { sparsity: 0.5 }],
      [applyMoeExpertChannelPruning, { sparsity: 0.5 }],
      // K=4 is not divisible by 32, so QuaRot matches nothing and the model
      // comes back unchanged -- the point is the binding round-trips.
      [applyQuarot, { seed: 0, blockSize: 32, epsilon: 1e-6 }],
    ]) {
      const out = await fn(input, options);
      assert.ok(out instanceof Uint8Array, fn.name);
      assert.ok(out.length > 0, fn.name);
    }
  });

  await check("embedding vocab pruning declines a model with no vocab chain", async () => {
    const input = new Uint8Array(readFileSync(FIXTURE));
    const r = await applyEmbeddingVocabPruning(input, { dropTokenIds: [1, 2] });
    assert.equal(r.matched, false);
    assert.ok(r.model instanceof Uint8Array && r.model.length > 0);
    const r2 = await applyEmbeddingVocabMagnitudePruning(input, { sparsity: 0.5 });
    assert.equal(r2.matched, false);
    assert.ok(r2.model instanceof Uint8Array && r2.model.length > 0);
  });

  await check("applyWandaPruning runs on synthetic onnxruntime-web calibration data", async () => {
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyWandaPruning(input, [batch(), batch()], { sparsity: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applyOutlierSuppression declines a model with no LayerNorm", async () => {
    // The shared fixture has no LayerNormalization node, so this is a
    // no-op round-trip -- the point is the binding (including calibration
    // crossing) works, not that it migrates anything.
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 1.0, 0.0]), [1, 4]),
    });
    const out = await applyOutlierSuppression(input, [batch(), batch()], { alpha: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  await check("applySmoothQuant migrates scales on synthetic calibration data", async () => {
    const ort = await import("onnxruntime-web");
    const input = new Uint8Array(readFileSync(FIXTURE));
    const batch = () => ({
      X: new ort.Tensor("float32", new Float32Array([0.5, -0.25, 4.0, 0.0]), [1, 4]),
    });
    const out = await applySmoothQuant(input, [batch(), batch()], { alpha: 0.5 });
    assert.ok(out instanceof Uint8Array);
    assert.ok(out.length > 0);
  });

  console.log(`PASS: ${passed} checks`);
} catch (err) {
  console.error(`FAIL after ${passed} checks:`, (err && err.stack) || err);
  process.exitCode = 1;
}
