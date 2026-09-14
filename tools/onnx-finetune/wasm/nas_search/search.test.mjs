// Does a tiny NAS search loop actually distinguish architectures -- not
// just "every candidate's step graph runs", but "training each candidate on
// the same problem produces a real, measurable ranking, and the loop
// correctly picks the best one" -- when run through the *official*
// onnxruntime-web package (no custom WASM build, no onnxruntime.training).
//
// Skips (does not fail) if onnxruntime-web is not installed here, matching
// ../federated_lora/step_graph_runner.test.mjs's own convention -- `npm
// install` in this directory to run it for real. generate_nas_step_graphs.py
// needs onnxsim's own C++ extension built (see the repo's CLAUDE.md) -- this
// test does not attempt to build it, only to run it.
//
// Usage:
//   npm install && npm test
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { combinedScore, makeSyntheticBatch, searchArchitectures, softmax } from "./search.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPTS = join(HERE, "..", "..", "scripts");

let ort;
try {
  ort = await import("onnxruntime-web");
} catch {
  ort = null;
}

function runPython(script, args) {
  execFileSync("python3", [join(SCRIPTS, script), ...args], { stdio: "inherit" });
}

/** Reads every `<name>.step_graph.onnx` generate_nas_step_graphs.py wrote
 * into `dir`, alongside its own `.manifest.txt`/`.initial_state.bin`, into
 * the `{ name, paramCount, stepGraphBytes, manifestText, initialStateBuffer }`
 * shape `searchArchitectures` wants -- the Node/fs equivalent of a browser
 * page `fetch()`ing the same three files per candidate. */
function loadCandidates(dir) {
  const spacePath = join(dir, "search_space.json");
  const space = JSON.parse(readFileSync(spacePath, "utf8"));
  return space.candidates.map((c) => {
    const stepPath = join(dir, c.step_graph);
    // A Node Buffer's own .buffer is the underlying (possibly pooled,
    // larger) ArrayBuffer, not a slice sized to this file -- .slice() by
    // this Buffer's own byteOffset/byteLength is what actually recovers
    // just the bytes this file contains, the same fix
    // ../federated_lora/step_graph_runner.test.mjs's own trainOneClient
    // already applies to the same readFileSync(...).buffer pattern.
    const initialStateBytes = readFileSync(`${stepPath}.initial_state.bin`);
    return {
      name: c.name,
      paramCount: c.param_count,
      stepGraphBytes: new Uint8Array(readFileSync(stepPath)),
      manifestText: readFileSync(`${stepPath}.manifest.txt`, "utf8"),
      initialStateBuffer: initialStateBytes.buffer.slice(
        initialStateBytes.byteOffset,
        initialStateBytes.byteOffset + initialStateBytes.byteLength,
      ),
    };
  });
}

test(
  "NAS search trains every candidate and ranks a higher-capacity architecture above an under-capacity one",
  { skip: ort === null && "onnxruntime-web not installed -- npm install to run this test" },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "onnx-finetune-nas-search-"));
    const batchSize = 32, dim = 16, out = 8;
    runPython("generate_nas_step_graphs.py", [
      "-o", dir, "--batch-size", String(batchSize), "--dim", String(dim), "--out", String(out),
    ]);

    const files = readdirSync(dir);
    assert.ok(files.includes("search_space.json"), `search_space.json missing among ${files}`);
    const candidates = loadCandidates(dir);
    assert.ok(candidates.length >= 2, "expected at least two candidates in the search space");
    // Every candidate really is a different shape -- otherwise a ranking
    // test could pass by accident even if generate_nas_step_graphs.py
    // silently produced the same architecture under different names.
    const distinctParamCounts = new Set(candidates.map((c) => c.paramCount));
    assert.ok(distinctParamCounts.size > 1, "expected candidates of different sizes");

    const { batchInput, target } = makeSyntheticBatch(batchSize, dim, out, 0);
    const options = {
      lr: 0.05,
      numSteps: 300,
      // The wasm provider requested explicitly: WebGPU -- the Vulkan-backed
      // path -- is unreachable under plain Node (no navigator.gpu;
      // onnxruntime-web answers "[webgpu] backend not found"), so this
      // exercises the one EP that exists here while proving the
      // `executionProviders` option is threaded all the way through
      // (search.mjs -> StepGraphSession.create -> InferenceSession.create).
      // The repo's own real-browser WebGPU training demos
      // (scripts/convertmodel/test/webgpu_*.test.mjs) prove the same
      // step-graph op set trains on WebGPU.
      executionProviders: ["wasm"],
    };
    const { ranked, best } = await searchArchitectures(ort, candidates, batchInput, target, options);

    for (const result of ranked) {
      for (const loss of result.losses) {
        assert.ok(Number.isFinite(loss), `${result.name}: loss ${loss} is not finite`);
      }
      assert.ok(
        result.finalLoss < result.losses[0],
        `${result.name}: loss should decrease over training (${result.losses[0]} -> ${result.finalLoss})`,
      );
      // Real measured wall-clock latency, not a placeholder -- see
      // trainCandidate's own comment on why this can be a plain
      // performance.now() delta rather than a predicted/looked-up latency.
      assert.ok(Number.isFinite(result.meanStepMs), `${result.name}: meanStepMs ${result.meanStepMs} is not finite`);
      assert.ok(result.meanStepMs > 0, `${result.name}: meanStepMs should be positive, got ${result.meanStepMs}`);
      assert.ok(
        Math.abs(result.totalMs - result.meanStepMs * options.numSteps) < 1e-6,
        `${result.name}: totalMs should equal meanStepMs * numSteps`,
      );
    }

    // The smallest-capacity candidate ("tiny", hidden width 4 -- see
    // generate_nas_step_graphs.py's SEARCH_SPACE) is a strict
    // representational bottleneck for this problem: its output, for any
    // input, is confined to the (at most) 4-dimensional subspace its final
    // linear layer's 4 columns span, but the synthetic target is a
    // generically full-rank (rank 8) linear function of a 16-dimensional
    // input -- so no matter how long "tiny" trains, it cannot drive the
    // residual outside that subspace to zero, and must not win the search
    // regardless of how the other (wider, unbottlenecked) candidates happen
    // to rank against each other.
    const tiny = ranked.find((r) => r.name === "tiny");
    assert.ok(tiny, "expected a 'tiny' candidate in the search space");
    assert.notEqual(best.name, "tiny", "the representationally-bottlenecked candidate must not win");
    assert.ok(
      best.finalLoss < 0.5 * tiny.finalLoss,
      `best candidate (${best.name}: ${best.finalLoss}) should clearly beat tiny (${tiny.finalLoss})`,
    );
  },
);

test("makeSyntheticBatch is reproducible for a fixed seed", () => {
  const a = makeSyntheticBatch(4, 3, 2, 42);
  const b = makeSyntheticBatch(4, 3, 2, 42);
  assert.deepEqual(Array.from(a.batchInput), Array.from(b.batchInput));
  assert.deepEqual(Array.from(a.target), Array.from(b.target));
  assert.equal(a.batchInput.length, 4 * 3);
  assert.equal(a.target.length, 4 * 2);
});

test("makeSyntheticBatch's inputScale/normalizeByDim options apply as documented", () => {
  const plain = makeSyntheticBatch(4, 3, 2, 7);
  const scaled = makeSyntheticBatch(4, 3, 2, 7, { inputScale: 5 });
  for (let i = 0; i < plain.batchInput.length; ++i) {
    assert.ok(
      Math.abs(scaled.batchInput[i] - 5 * plain.batchInput[i]) < 1e-4,
      `entry ${i}: ${scaled.batchInput[i]} should be 5x ${plain.batchInput[i]}`,
    );
  }
  const normalized = makeSyntheticBatch(4, 3, 2, 7);
  const unnormalized = makeSyntheticBatch(4, 3, 2, 7, { normalizeByDim: false });
  // Same seed draws the same batchInput either way -- normalizeByDim only
  // changes the target formula, not which random numbers get drawn.
  assert.deepEqual(Array.from(normalized.batchInput), Array.from(unnormalized.batchInput));
  // The two targets differ (by a factor of sqrt(dim) on the linear part,
  // obscured by an independent noise draw per entry -- not worth an exact
  // ratio check here, just that turning the flag off actually changes
  // something).
  assert.notDeepEqual(Array.from(normalized.target), Array.from(unnormalized.target));
});

test("combinedScore is loss-only when latencyWeight is 0, and penalizes latency otherwise", () => {
  const fast = { finalLoss: 1.0, meanStepMs: 1 };
  const slow = { finalLoss: 0.9, meanStepMs: 100 };
  assert.equal(combinedScore(fast), 1.0);
  assert.equal(combinedScore(slow), 0.9);
  assert.ok(combinedScore(slow, 0) < combinedScore(fast, 0), "with no latency weight, lower loss should still win");
  assert.ok(
    combinedScore(fast, 0.1) < combinedScore(slow, 0.1),
    "with a large enough latency weight, the faster candidate should win despite the higher loss",
  );
});

test("softmax sums to 1 and preserves ordering", () => {
  const weights = softmax([1, 2, 3]);
  const sum = weights.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `softmax should sum to 1, got ${sum}`);
  assert.ok(weights[0] < weights[1] && weights[1] < weights[2], "softmax should preserve input ordering");
  assert.deepEqual(softmax([5, 5, 5]).map((w) => Math.round(w * 1000) / 1000), [1 / 3, 1 / 3, 1 / 3].map((w) => Math.round(w * 1000) / 1000));
});
