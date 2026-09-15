// Runs the WebGPU programs onnxsim.webgpu_tinygrad_codegen actually
// *generates* (not hand-written, like webgpu_kernel_dispatcher.test.mjs's
// Add kernel) against a real WebGPU device -- the first time a
// tinygrad-rendered kernel executes on a GPU at all, rather than only being
// checked offline against onnx.reference.ReferenceEvaluator on tinygrad's
// own CPU device (see tests/test_webgpu_tinygrad_codegen.py, the other half
// of this generator's own verification story per its module docstring).
//
// webgpu_tinygrad_conv3d.onnx / webgpu_tinygrad_resize.onnx (see
// make_webgpu_tinygrad_codegen_fixture.py) each carry one node with a
// tinygrad-generated WebGPU *program* (one or more steps -- Resize
// schedules as two kernels sharing a scratch buffer, see
// onnxsim/webgpu_kernel_metadata.py's own docstring for why the schema
// supports that) attached via generate_conv_kernel/generate_resize_kernel.
// webgpu_tinygrad_codegen_fixture.json records the exact input values used
// and the expected output (from onnx.reference.ReferenceEvaluator running
// the real node), computed once in Python at fixture-generation time -- this
// test only fetches the .onnx file's bytes and the fixture JSON; it never
// needs Python or tinygrad itself, same as the other webgpu_*.test.mjs files
// here need no Python at test time.
//
// For each fixture, this:
//   1. extracts the program via onnx_node_metadata.mjs's
//      readWebgpuKernelSpecs -- the same function a real runtime would use.
//   2. uploads the fixture's own input tensor(s) to the GPU.
//   3. dispatches the program via webgpu_kernel_dispatcher.mjs
//      (dispatchWebgpuProgram runs every step in order, allocating its own
//      intermediate/constant buffers -- nothing here has to know about
//      those).
//   4. reads the output tensor back and checks it against the fixture's
//      expected values -- proving the *generated* WGSL actually ran on the
//      GPU and computed the right thing, not just that it parsed.
//
// Requires the "playwright" package and a Chromium binary -- same
// requirement as webgpu_kernel_dispatcher.test.mjs (see that file's own
// comment for the exact SwiftShader/software-Vulkan caveat).
//
// Usage:
//   npx playwright install chromium   # once
//   pip install onnx numpy tinygrad==0.14.0
//   python3 make_webgpu_tinygrad_codegen_fixture.py   # regenerate fixtures if needed
//   node test/webgpu_tinygrad_codegen.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_tinygrad_codegen_fixture.json"), "utf8"));

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu tinygrad codegen</title>");
      return;
    }
    const filePath = join(ROOT, reqPath);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found: " + reqPath);
        return;
      }
      const type = {
        ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
        ".json": "application/json", ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
      }[path.extname(filePath)] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": type });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, () => resolve(server));
  });
}

// Runs inside the real browser page via page.evaluate -- no access to this
// file's own scope, only what's passed in and what it can import()/fetch().
async function runCaseInPage({ port, file, nodeName, outputName, inputs, expectedLength }) {
  const base = `http://localhost:${port}`;
  const { readWebgpuKernelSpecs } = await import(`${base}/onnx_node_metadata.mjs`);
  const { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer } = await import(
    `${base}/webgpu_kernel_dispatcher.mjs`
  );

  const modelBytes = new Uint8Array(await fetch(`${base}/test/${file}`).then((r) => r.arrayBuffer()));
  const specs = readWebgpuKernelSpecs(modelBytes);
  if (!specs.has(nodeName)) {
    return { ok: false, error: `no kernel spec found for node ${nodeName}` };
  }
  const spec = specs.get(nodeName);

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const buffersByTensor = new Map();
  for (const [name, { data }] of Object.entries(inputs)) {
    buffersByTensor.set(name, createStorageBuffer(device, Float32Array.from(data)));
  }
  buffersByTensor.set(outputName, createStorageBuffer(device, new Float32Array(expectedLength)));

  await dispatchWebgpuProgram(device, spec, buffersByTensor);

  const actual = await readBackFloat32Buffer(device, buffersByTensor.get(outputName), expectedLength);
  return { ok: true, actual: Array.from(actual) };
}

async function main() {
  console.log("WebGPU tinygrad-generated kernel dispatch check\n");

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // Verified sufficient on its own in this repo's own dev sandbox (via
    // SwiftShader's software Vulkan path); see webgpu_hf_demo.test.mjs's own
    // comment. A real GPU, as CI runners with one have, needs nothing more.
    args: ["--enable-unsafe-webgpu"],
  });

  try {
    for (const [caseName, fixture] of Object.entries(MANIFEST)) {
      const page = await browser.newPage();
      await page.goto(`http://localhost:${port}/`);
      const result = await page.evaluate(runCaseInPage, {
        port,
        file: fixture.file,
        nodeName: fixture.nodeName,
        outputName: fixture.outputName,
        inputs: fixture.inputs,
        expectedLength: fixture.expectedOutput.data.length,
      });
      await page.close();

      await check(`${caseName}: kernel spec found and dispatch ran without throwing`, () => {
        assert.ok(result.ok, `dispatch failed: ${result.error}`);
      });

      const expected = fixture.expectedOutput.data;
      let maxAbsDiff = 0;
      for (let i = 0; i < expected.length; i++) {
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(result.actual[i] - expected[i]));
      }
      await check(`${caseName}: GPU output matches onnx.reference.ReferenceEvaluator`, () => {
        assert.ok(
          maxAbsDiff < 1e-3,
          `max abs diff ${maxAbsDiff} too large -- sample actual ${JSON.stringify(result.actual.slice(0, 4))} ` +
            `vs expected ${JSON.stringify(expected.slice(0, 4))}`,
        );
      });
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`\nwebgpu tinygrad codegen dispatcher: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
