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
import { simplify, versions } from "../index.mjs";

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

  console.log(`PASS: ${passed} checks`);
} catch (err) {
  console.error(`FAIL after ${passed} checks:`, (err && err.stack) || err);
  process.exitCode = 1;
}
