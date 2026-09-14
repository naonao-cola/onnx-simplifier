// tools/onnx-finetune/wasm/nas_search's NAS search loop, trained end to end
// on a real browser's WebGPU backend -- verified the same way
// webgpu_train_loop_demo.test.mjs verifies onnxsim.compile_training's own
// step graphs: not "attempted and reported" the way
// nas_search/search.test.mjs treats webgpu under plain Node (which has no
// navigator.gpu at all -- see that file's own comment), but REQUIRED, via a
// real Chromium page Playwright drives specifically so WebGPU is reachable.
//
// What this proves, and what it does not:
//   * Every op tools/onnx-finetune/scripts/generate_nas_step_graphs.py's
//     candidates are built from (graph_grad.build_backward's own
//     BACKWARD_OPS/qat_graph.EP_FRIENDLY_OPS allowlists) actually placed on
//     WebGPU (JsExecutionProvider), for a real candidate step graph -- not
//     just "the allowlist says these ops are covered".
//   * The search loop (nas_search/search.mjs's searchArchitectures, run
//     completely unmodified -- this file adds no WebGPU-specific code of
//     its own to that module) reaches the same conclusion on WebGPU that
//     search.test.mjs already established on the wasm (CPU) execution
//     provider under plain Node: the representationally-bottlenecked "tiny"
//     candidate does not win.
//   * It does NOT re-derive a numeric CPU reference trajectory the way
//     webgpu_train_loop_demo.test.mjs does (that test's fixture ships one,
//     computed once in Python); this one instead trains every candidate a
//     second time on the wasm EP inside the same page and checks the two
//     runs agree on which candidate wins, which needs no separate fixture
//     and still catches a WebGPU-only numerical breakage.
//
// Requires the "playwright" package and a Chromium binary -- both already
// declared in this directory's own package.json (`npm install`). This
// sandbox has Chromium pre-installed under /opt/pw-browsers rather than via
// `npx playwright install`, so this file resolves the executable from there
// when present and falls back to Playwright's own default resolution (e.g.
// in CI, which runs `npx playwright install chromium` --
// .github/workflows/convertmodel-webgpu-demo.yml) otherwise.
//
// No onnxsim wheel build here, same as webgpu_train_loop_demo.test.mjs's own
// choice and for the same reason (that file's own comment): the candidates
// under nas_search_candidates/ are pre-generated, committed fixtures --
// `python3 ../../../tools/onnx-finetune/scripts/generate_nas_step_graphs.py
// -o test/nas_search_candidates --batch-size 32 --dim 16 --out 8`, run once
// with onnxsim's own C++ extension built (see the repo's CLAUDE.md) and the
// result checked in, exactly like sibling fixtures
// step_train_loop_demo.onnx/.json.
//
// Usage:
//   npm install && node test/webgpu_nas_search.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { existsSync, readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const CONVERTMODEL_ROOT = join(HERE, "..");
const WASM_ROOT = join(HERE, "..", "..", "..", "tools", "onnx-finetune", "wasm");
const CANDIDATES_DIR = join(HERE, "nas_search_candidates");

const BATCH_SIZE = 32, DIM = 16, OUT = 8;
const LR = 0.05, NUM_STEPS = 300;

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Serves three separate directory trees under one port: this test's own
// scripts/convertmodel checkout (for node_modules/onnxruntime-web), the
// nas_search/federated_lora runner modules, and the freshly-generated
// candidates in a temp dir -- so the in-page dynamic import()s and fetch()es
// below can resolve all three by plain relative/absolute URL, the same way
// webgpu_train_loop_demo.test.mjs's own serveConvertmodelDir serves just the
// one tree it needs.
function serveRoots(roots) {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu nas search demo</title>");
      return;
    }
    const [, prefix, ...rest] = reqPath.split("/");
    const root = roots[prefix];
    if (!root) {
      res.writeHead(404);
      res.end("no such root: " + prefix);
      return;
    }
    const filePath = join(root, ...rest);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found: " + reqPath);
        return;
      }
      const type = {
        ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
        ".json": "application/json", ".onnx": "application/octet-stream",
        ".wasm": "application/wasm", ".bin": "application/octet-stream", ".txt": "text/plain",
      }[path.extname(filePath)] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": type });
      res.end(data);
    });
  });
  return new Promise((resolve) => server.listen(0, () => resolve(server)));
}

// Everything below runs inside the real browser page, via page.evaluate --
// it has no access to this file's own scope, only what it's passed and what
// it can import()/fetch() from the server above.
async function runInPage({ port, batchSize, dim, out, lr, numSteps }) {
  const base = `http://localhost:${port}`;
  const ortMod = await import(`${base}/cm/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = `${base}/cm/node_modules/onnxruntime-web/dist/`;

  const search = await import(`${base}/wasm/nas_search/search.mjs`);

  const space = await fetch(`${base}/candidates/search_space.json`).then((r) => r.json());
  const loadCandidate = async (c) => {
    const stepUrl = `${base}/candidates/${c.step_graph}`;
    const [stepGraphBytes, manifestText, initialStateBuffer] = await Promise.all([
      fetch(stepUrl).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
      fetch(`${stepUrl}.manifest.txt`).then((r) => r.text()),
      fetch(`${stepUrl}.initial_state.bin`).then((r) => r.arrayBuffer()),
    ]);
    return { name: c.name, paramCount: c.param_count, numNodes: c.num_nodes, stepGraphBytes, manifestText, initialStateBuffer };
  };
  const candidates = await Promise.all(space.candidates.map(loadCandidate));

  const { batchInput, target } = search.makeSyntheticBatch(batchSize, dim, out, 0);
  const options = { lr, numSteps };
  const webgpu = await search.searchArchitectures(ort, candidates, batchInput, target, {
    ...options, executionProviders: ["webgpu", "wasm"],
  });
  const wasm = await search.searchArchitectures(ort, candidates, batchInput, target, {
    ...options, executionProviders: ["wasm"],
  });

  // The most structurally complex candidate's own placement is checked
  // separately, by runPlacementCheckInPage below (a plain, unoptimized,
  // verbosely-logged session -- webgpu/wasm above intentionally leave
  // optimization on and logging off, since that's the realistic way a
  // search loop would actually be run). Every candidate shares the same
  // op-type vocabulary regardless of width/depth (see
  // generate_nas_step_graphs.py's own comment on `ops`/`num_nodes`), so
  // checking any single one already covers every op type the whole search
  // space uses -- num_nodes just picks a deterministic, meaningfully
  // "biggest" one rather than an arbitrary first entry.
  const mostComplex = candidates.reduce((a, b) => (b.numNodes > a.numNodes ? b : a));

  return { webgpu, wasm, mostComplexName: mostComplex.name, candidateCount: candidates.length };
}

// Same parser step_graph_ep.test.mjs/webgpu_hf_demo.test.mjs/
// webgpu_train_loop_demo.test.mjs all use, duplicated rather than imported
// per that file's own comment on why: no shared module needs it on the Node
// side of a real browser run.
function parsePlacements(lines) {
  const placements = {};
  let current = null;
  for (const raw of lines) {
    const line = raw.replace(/^.*VerifyEachNodeIsAssignedToAnEp\]\s?/, "");
    const header = line.match(/(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/);
    if (header) {
      const all = header[1] === "All nodes";
      current = placements[header[2]] || { count: 0, ops: all ? null : new Set() };
      current.count += Number(header[3]);
      if (all) current.ops = null;
      placements[header[2]] = current;
      continue;
    }
    const node = current && current.ops && line.match(/^\s*([A-Za-z][\w.]*)\s*\(/);
    if (node) current.ops.add(node[1]);
  }
  return placements;
}

// Runs one candidate's step graph directly (bypassing StepGraphSession, so
// graphOptimizationLevel/log options -- which that wrapper does not forward
// -- reach InferenceSession.create) purely to capture ORT's own per-node EP
// placement report.
async function runPlacementCheckInPage({ port, stepGraphUrl, manifestUrl, initialStateUrl, batchSize, dim, out, lr }) {
  const base = `http://localhost:${port}`;
  const ortMod = await import(`${base}/cm/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = `${base}/cm/node_modules/onnxruntime-web/dist/`;
  const runner = await import(`${base}/wasm/federated_lora/step_graph_runner.mjs`);

  const [stepGraphBytes, manifestText, initialStateBuffer] = await Promise.all([
    fetch(stepGraphUrl).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
    fetch(manifestUrl).then((r) => r.text()),
    fetch(initialStateUrl).then((r) => r.arrayBuffer()),
  ]);
  const manifest = runner.parseManifest(manifestText);
  const state = runner.loadInitialState(manifest, initialStateBuffer);

  const session = await ort.InferenceSession.create(stepGraphBytes, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "disabled",
    logSeverityLevel: 0,
    logVerbosityLevel: 4,
  });
  const T = ort.Tensor;
  const { mCorrection, vCorrection } = runner.adamBiasCorrections(0);
  const feeds = {
    [manifest.inputName]: new T("float32", new Float32Array(batchSize * dim), manifest.inputShape),
    [manifest.teacherName]: new T("float32", new Float32Array(batchSize * out), manifest.teacherShape),
    [manifest.lrName]: new T("float32", new Float32Array([lr]), []),
    m_correction: new T("float32", new Float32Array([mCorrection]), []),
    v_correction: new T("float32", new Float32Array([vCorrection]), []),
  };
  for (const { input, shape } of manifest.state) feeds[input] = new T("float32", state[input], shape);
  const results = await session.run(feeds);
  const loss = Number(results[manifest.lossName].data[0]);
  await session.release?.();
  return { loss };
}

async function main() {
  const space = JSON.parse(readFileSync(join(CANDIDATES_DIR, "search_space.json"), "utf8"));
  assert.equal(space.batch_size, BATCH_SIZE);
  assert.equal(space.dim, DIM);
  assert.equal(space.out, OUT);
  console.log(`WebGPU NAS search demo -- ${space.candidates.length} candidates, ${NUM_STEPS} steps each\n`);

  const server = await serveRoots({ cm: CONVERTMODEL_ROOT, wasm: WASM_ROOT, candidates: CANDIDATES_DIR });
  const port = server.address().port;

  // This sandbox ships Chromium pre-installed under /opt/pw-browsers rather
  // than via `npx playwright install` (see this file's own header comment);
  // resolve it explicitly when present, and let Playwright fall back to its
  // own default resolution (real CI's `npx playwright install chromium`)
  // otherwise.
  const SANDBOX_CHROMIUM = "/opt/pw-browsers/chromium";
  const launchOptions = {
    headless: true,
    // --enable-unsafe-webgpu: verified sufficient on its own in this repo's
    // own dev sandbox (via SwiftShader's software Vulkan path); see
    // webgpu_hf_demo.test.mjs's own comment. A real GPU, as a CI runner
    // with one would have, needs nothing more. --no-sandbox: this sandbox
    // runs as a user Chromium's own sandbox refuses under, same as the
    // other webgpu_*.test.mjs files would need if run here.
    args: existsSync(SANDBOX_CHROMIUM)
      ? ["--enable-unsafe-webgpu", "--no-sandbox"]
      : ["--enable-unsafe-webgpu"],
  };
  if (existsSync(SANDBOX_CHROMIUM)) launchOptions.executablePath = SANDBOX_CHROMIUM;

  const browser = await chromium.launch(launchOptions);
  const logLines = [];
  let result, placement;
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (/onnxruntime/i.test(text)) logLines.push(text);
    });
    // A real page, not about:blank -- WebGPU's adapter is unreachable from
    // about:blank in this sandbox (verified while writing this test).
    await page.goto(`http://localhost:${port}/`);
    result = await page.evaluate(runInPage, { port, batchSize: BATCH_SIZE, dim: DIM, out: OUT, lr: LR, numSteps: NUM_STEPS });

    const mostComplex = space.candidates.reduce((a, b) => (b.num_nodes > a.num_nodes ? b : a));
    const stepUrl = `http://localhost:${port}/candidates/${mostComplex.step_graph}`;
    placement = await page.evaluate(runPlacementCheckInPage, {
      port, stepGraphUrl: stepUrl, manifestUrl: `${stepUrl}.manifest.txt`, initialStateUrl: `${stepUrl}.initial_state.bin`,
      batchSize: BATCH_SIZE, dim: DIM, out: OUT, lr: LR,
    });
  } finally {
    await browser.close();
    server.close();
  }

  await check(`all ${result.candidateCount} candidates loaded`, () => {
    assert.equal(result.candidateCount, space.candidates.length);
  });

  for (const [label, run] of [["webgpu", result.webgpu], ["wasm", result.wasm]]) {
    await check(`${label}: every candidate's loss is finite and decreases`, () => {
      for (const r of run.ranked) {
        for (const loss of r.losses) assert.ok(Number.isFinite(loss), `${label}/${r.name}: loss ${loss} not finite`);
        assert.ok(r.finalLoss < r.losses[0], `${label}/${r.name}: loss did not decrease (${r.losses[0]} -> ${r.finalLoss})`);
      }
    });
    await check(`${label}: the representationally-bottlenecked 'tiny' candidate does not win`, () => {
      assert.notEqual(run.best.name, "tiny", `${label} picked 'tiny' as the winner`);
    });
  }

  await check("webgpu and wasm agree on which candidate wins", () => {
    assert.equal(result.webgpu.best.name, result.wasm.best.name,
      `webgpu picked ${result.webgpu.best.name}, wasm picked ${result.wasm.best.name}`);
  });

  await check(`most structurally complex candidate (${result.mostComplexName}) trains with a finite loss on WebGPU alone`, () => {
    assert.ok(Number.isFinite(placement.loss), `loss ${placement.loss} is not finite`);
  });

  const placements = parsePlacements(logLines);
  await check("every op in the most structurally complex candidate's step graph ran on WebGPU (JsExecutionProvider)", () => {
    const mostComplexCandidate = space.candidates.find((c) => c.name === result.mostComplexName);
    const gpu = placements.JsExecutionProvider;
    assert.ok(gpu, `no JsExecutionProvider placement reported at all -- captured ${logLines.length} log lines`);
    const gpuOps = gpu.ops === null ? new Set(mostComplexCandidate.ops) : gpu.ops;
    const missing = mostComplexCandidate.ops.filter((op) => !gpuOps.has(op));
    assert.deepEqual(missing, [],
      `these ops never showed up under JsExecutionProvider: ${missing.join(", ")} -- ` +
        `full placement report: ${JSON.stringify(placements, (k, v) => (v instanceof Set ? [...v] : v))}`);
    for (const [provider, info] of Object.entries(placements)) {
      if (provider === "JsExecutionProvider") continue;
      const strayed = info.ops && info.ops.size && [...info.ops].filter((op) => mostComplexCandidate.ops.includes(op));
      assert.ok(!strayed || strayed.length === 0, `${strayed && strayed.join(", ")} ran on ${provider} instead of WebGPU`);
    }
  });

  console.log(`\nwebgpu nas search demo: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
