// Round-trips a model through onnxsim_extract_graph (ModelProto bytes -> ONNX
// text) and back through onnxsim_parse_graph (text -> ModelProto bytes) --
// the two bindings behind the "Parse a text graph" panel's "Parse graph" and
// "Extract from loaded model" buttons (see debug_tools.mjs). Checks that the
// extracted text actually names the fixture's op/value names, and that
// reparsing it reproduces the same graph inputs/outputs onnxsim_parse_graph
// produced from the original text.
//
// Needs the wasm module built and staged next to the page
// (scripts/convertmodel/onnxsim.js + onnxsim.wasm, e.g. `./build_wasm.sh` then
// `cp build-wasm-node-OFF/onnxsim.* scripts/convertmodel`, which is what
// .github/workflows/static.yml does). Without it the test SKIPS, since the
// convertmodel test job builds no wasm; EXTRACT_REQUIRE_WASM=1 turns that skip
// into a failure for a job that does build it.
//
// Usage:
//   node test/extract_graph.test.mjs
//   EXTRACT_REQUIRE_WASM=1 node test/extract_graph.test.mjs   # skip => failure

import assert from "node:assert/strict";
import { copyFileSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { readShapes } from "../shapes.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const MODULE_JS = join(HERE, "..", "onnxsim.js");
const MODULE_WASM = join(HERE, "..", "onnxsim.wasm");

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Same loader as qat_step_graph.test.mjs / lora_step_graph.test.mjs -- see
// either file's own comment for why onnxsim.js is staged as .cjs here.
function loadRuntime() {
  const dir = mkdtempSync(join(tmpdir(), "onnxsim-extract-"));
  const cjs = join(dir, "onnxsim.cjs");
  copyFileSync(MODULE_JS, cjs);
  const require = createRequire(import.meta.url);
  const createOnnxsim = require(cjs);
  return {
    dir,
    module: createOnnxsim({
      locateFile: (path) => (path.endsWith(".wasm") ? MODULE_WASM : join(HERE, "..", path)),
      print: (s) => console.log("    | " + s),
      printErr: (s) => console.log("    ! " + s),
    }),
  };
}

function copyBytes(view) {
  return new Uint8Array(view).slice();
}

function names(valueInfos) {
  return valueInfos.map((v) => v.name);
}

const FIXTURE_TEXT = `<
   ir_version: 8,
   opset_import: ["" : 13]
>
agraph (float[N] X) => (float[N] Y) {
   T = Relu(X)
   Y = Sqrt(T)
}`;

async function main() {
  if (!existsSync(MODULE_JS) || !existsSync(MODULE_WASM)) {
    const why =
      `scripts/convertmodel/onnxsim.{js,wasm} not found -- build the wasm module ` +
      `(./build_wasm.sh) and copy it next to the page to run this test`;
    if (process.env.EXTRACT_REQUIRE_WASM) {
      console.error("FAIL:", why);
      process.exit(1);
    }
    console.log("  skip - " + why);
    return;
  }

  const loaded = loadRuntime();
  let runtime;
  try {
    runtime = await loaded.module;
  } finally {
    rmSync(loaded.dir, { recursive: true, force: true });
  }

  const parsed = runtime.onnxsim_parse_graph(FIXTURE_TEXT);
  assert.equal(parsed.error, undefined, `fixture did not parse: ${parsed.error}`);
  const originalModel = copyBytes(parsed.model);
  const originalShapes = readShapes(originalModel);

  const extracted = runtime.onnxsim_extract_graph(originalModel);

  await check("extraction succeeds and names the fixture's ops and values", () => {
    assert.equal(extracted.error, undefined, `extraction failed: ${extracted.error}`);
    assert.equal(typeof extracted.text, "string");
    for (const token of ["Relu", "Sqrt", "X", "Y", "agraph"]) {
      assert.ok(extracted.text.includes(token), `extracted text is missing "${token}":\n${extracted.text}`);
    }
  });

  await check("the extracted text reparses to the same graph inputs/outputs", () => {
    const reparsed = runtime.onnxsim_parse_graph(extracted.text);
    assert.equal(reparsed.error, undefined, `extracted text did not reparse: ${reparsed.error}`);
    const reparsedShapes = readShapes(copyBytes(reparsed.model));
    assert.deepEqual(names(reparsedShapes.inputs), names(originalShapes.inputs));
    assert.deepEqual(names(reparsedShapes.outputs), names(originalShapes.outputs));
  });

  await check("bytes that are not a ModelProto are refused with an error", () => {
    const junk = new Uint8Array([1, 2, 3, 255, 255, 255]);
    const res = runtime.onnxsim_extract_graph(junk);
    assert.equal(typeof res.error, "string");
    assert.equal(res.text, undefined);
  });

  console.log(`\nextract graph: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
