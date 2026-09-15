// Proves webgpu_custom_kernel_runtime.mjs actually works end to end: a real
// onnxruntime-web WebGPU session runs the "pre" half of a model
// (webgpu_custom_kernel_runtime_pre.onnx, a Relu), its GPU-buffer output is
// spliced -- with no CPU round-trip -- into a tinygrad-generated WebGPU
// Conv3D program (dispatched via webgpu_kernel_dispatcher.mjs), and *that*
// result is spliced into a second real onnxruntime-web WebGPU session (the
// "post" half, another Relu), producing the model's real final output.
//
// webgpu_custom_kernel_runtime_fixture.json (see
// make_webgpu_custom_kernel_runtime_fixture.py) carries the pre/post model
// files, the excised Conv node's own WebgpuKernelSpec (from
// onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel), its weight
// initializer's concrete values (an "extra" input split_around_node leaves
// for the caller to source -- see that module's own docstring), and the
// expected final output from onnx.reference.ReferenceEvaluator running the
// *original*, unsplit model -- so this checks the whole spliced pipeline
// against the same ground truth the standalone codegen tests use, not just
// "it didn't throw".
//
// Requires the "playwright" package and a Chromium binary -- same
// requirement as the other webgpu_*.test.mjs files here.
//
// Usage:
//   npx playwright install chromium   # once
//   pip install onnx numpy tinygrad==0.14.0
//   python3 make_webgpu_custom_kernel_runtime_fixture.py   # regenerate fixtures if needed
//   node test/webgpu_custom_kernel_runtime.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_custom_kernel_runtime_fixture.json"), "utf8"));

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
      res.end("<!doctype html><title>webgpu custom kernel runtime</title>");
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
async function runInPage({ port, manifest }) {
  const base = `http://localhost:${port}`;
  try {
    const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
    const ort = ortMod.default ?? ortMod;
    ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;

    const { runOnnxModelWithCustomKernel } = await import(`${base}/webgpu_custom_kernel_runtime.mjs`);

    const preModelBytes = new Uint8Array(await fetch(`${base}/test/${manifest.preFile}`).then((r) => r.arrayBuffer()));
    const postModelBytes = new Uint8Array(
      await fetch(`${base}/test/${manifest.postFile}`).then((r) => r.arrayBuffer()),
    );

    const { name: inputName, shape: inputShape, data: inputData } = manifest.graphInput;
    const feeds = { [inputName]: new ort.Tensor("float32", Float32Array.from(inputData), inputShape) };

    const extraInputs = {};
    for (const [name, { data }] of Object.entries(manifest.extraInputs)) {
      extraInputs[name] = data;
    }

    const outputs = await runOnnxModelWithCustomKernel({
      ort,
      preModelBytes,
      postModelBytes,
      spec: manifest.spec,
      feeds,
      extraInputs,
      nodeOutputs: manifest.nodeOutputs,
      finalOutputNames: [manifest.finalOutputName],
    });

    const finalTensor = outputs[manifest.finalOutputName];
    return { ok: true, data: Array.from(finalTensor.data), dims: finalTensor.dims };
  } catch (e) {
    return { ok: false, error: e && e.stack ? e.stack : String(e) };
  }
}

async function main() {
  console.log("WebGPU custom-kernel-runtime end-to-end check\n");

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // Verified sufficient on its own in this repo's own dev sandbox (via
    // SwiftShader's software Vulkan path); see webgpu_hf_demo.test.mjs's own
    // comment. A real GPU, as CI runners with one have, needs nothing more.
    args: ["--enable-unsafe-webgpu"],
  });

  let result;
  try {
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/`);
    result = await page.evaluate(runInPage, { port, manifest: MANIFEST });
    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  await check("pipeline ran without throwing", () => {
    assert.ok(result.ok, `pipeline failed: ${result.error}`);
  });

  const expected = MANIFEST.expectedOutput;
  await check("final output shape matches onnx.reference.ReferenceEvaluator", () => {
    assert.deepEqual(result.dims, expected.shape);
  });

  await check("final output values match onnx.reference.ReferenceEvaluator", () => {
    let maxAbsDiff = 0;
    for (let i = 0; i < expected.data.length; i++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(result.data[i] - expected.data[i]));
    }
    assert.ok(
      maxAbsDiff < 1e-3,
      `max abs diff ${maxAbsDiff} too large -- actual ${JSON.stringify(result.data)} vs expected ${JSON.stringify(expected.data)}`,
    );
  });

  console.log(`\nwebgpu custom kernel runtime: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
