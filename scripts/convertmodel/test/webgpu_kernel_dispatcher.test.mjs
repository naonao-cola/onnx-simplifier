// Runs a real custom WebGPU kernel, read straight out of an .onnx model's
// own node metadata, against a real WebGPU device -- the actual "execute a
// custom WebGPU kernel from the model metadata" capability
// onnxsim/webgpu_kernel_metadata.py + onnx_node_metadata.mjs +
// webgpu_kernel_dispatcher.mjs together provide, exercised end to end for
// the first time here (the other two are otherwise only tested against
// plain data in Node, never against a live GPU).
//
// webgpu_kernel_add.onnx (see make_webgpu_kernel_fixture.py) carries one
// Add node named "add_node" with a hand-written WGSL elementwise-add kernel
// attached via attach_webgpu_kernel(). This test:
//   1. fetches the model bytes and extracts that kernel spec via
//      onnx_node_metadata.mjs's readWebgpuKernelSpecs -- the same function a
//      real runtime would use, not a shortcut around it.
//   2. uploads two random Float32Array buffers to the GPU.
//   3. dispatches the kernel via webgpu_kernel_dispatcher.mjs.
//   4. reads the output buffer back and checks it against a plain JS a[i]+b[i]
//      reference -- proving the WGSL actually ran on the GPU and did the
//      right thing, not just that dispatch didn't throw.
//
// Requires the "playwright" package and a Chromium binary -- same
// requirement as webgpu_attention_placement.test.mjs (see that file's own
// comment for the exact SwiftShader/software-Vulkan caveat).
//
// Usage:
//   npx playwright install chromium   # once
//   python3 make_webgpu_kernel_fixture.py   # regenerate fixtures if needed
//   node test/webgpu_kernel_dispatcher.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_kernel_fixture.json"), "utf8"));

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
      res.end("<!doctype html><title>webgpu kernel dispatcher</title>");
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
async function runInPage({ port, file, nodeName, n }) {
  const base = `http://localhost:${port}`;
  const { readWebgpuKernelSpecs } = await import(`${base}/onnx_node_metadata.mjs`);
  const { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer, supportsWebgpuProfiling } = await import(
    `${base}/webgpu_kernel_dispatcher.mjs`
  );

  const modelBytes = new Uint8Array(await fetch(`${base}/test/${file}`).then((r) => r.arrayBuffer()));
  const specs = readWebgpuKernelSpecs(modelBytes);
  if (!specs.has(nodeName)) {
    return { ok: false, error: `no kernel spec found for node ${nodeName}` };
  }
  const spec = specs.get(nodeName);

  const adapter = await navigator.gpu.requestAdapter();
  // Request "timestamp-query" up front (only if the adapter actually offers
  // it) so the profiling check below gets a real answer either way, rather
  // than always seeing "unsupported" just because nobody asked for the
  // feature at device-creation time -- a device's feature set can't be
  // grown after the fact.
  const wantTimestampQuery = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: wantTimestampQuery ? ["timestamp-query"] : [],
  });

  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = Math.sin(i);
    b[i] = Math.cos(i) * 2;
  }
  const expected = new Float32Array(n);
  for (let i = 0; i < n; i++) expected[i] = a[i] + b[i];

  const bufferA = createStorageBuffer(device, a);
  const bufferB = createStorageBuffer(device, b);
  const bufferC = createStorageBuffer(device, new Float32Array(n)); // zero-initialized output
  const buffers = new Map([
    ["a", bufferA],
    ["b", bufferB],
    ["c", bufferC],
  ]);

  const { timings } = await dispatchWebgpuProgram(device, spec, buffers, { profile: true });

  const actual = await readBackFloat32Buffer(device, bufferC, n);
  let maxAbsDiff = 0;
  for (let i = 0; i < n; i++) {
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual[i] - expected[i]));
  }
  return {
    ok: true,
    maxAbsDiff,
    sample: [actual[0], actual[1], actual[n - 1]],
    deviceSupportsProfiling: supportsWebgpuProfiling(device),
    timings,
  };
}

async function main() {
  console.log(`WebGPU custom kernel dispatch check -- node: ${MANIFEST.nodeName}\n`);

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
    result = await page.evaluate(runInPage, {
      port,
      file: MANIFEST.file,
      nodeName: MANIFEST.nodeName,
      n: MANIFEST.n,
    });
    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  await check("kernel spec found and dispatch ran without throwing", () => {
    assert.ok(result.ok, `dispatch failed: ${result.error}`);
  });

  await check("GPU output matches the CPU a[i]+b[i] reference", () => {
    assert.ok(
      result.maxAbsDiff < 1e-5,
      `max abs diff ${result.maxAbsDiff} too large -- sample output ${JSON.stringify(result.sample)}`,
    );
  });

  await check("profile: true reports a real per-step GPU duration when the device supports it", () => {
    if (!result.deviceSupportsProfiling) {
      console.log('    (adapter has no "timestamp-query" here -- skipping the timing assertion itself)');
      assert.equal(result.timings, null, "expected no timings from a device that can't report them");
      return;
    }
    assert.equal(result.timings.length, 1, "the Add fixture is a single-step program");
    const [timing] = result.timings;
    assert.equal(timing.index, 0);
    assert.equal(timing.entryPoint, MANIFEST.entryPoint);
    assert.ok(
      Number.isFinite(timing.durationNs) && timing.durationNs >= 0,
      `expected a non-negative finite duration, got ${timing.durationNs}`,
    );
    console.log(`    (GPU duration: ${timing.durationNs}ns)`);
  });

  console.log(`\nwebgpu kernel dispatcher: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
