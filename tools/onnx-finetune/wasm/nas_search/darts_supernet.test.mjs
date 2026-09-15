// Does a DARTS-style differentiable-architecture-search supernet
// (../../scripts/generate_darts_supernet_step_graph.py) actually learn an
// architecture preference -- not just "the step graph runs and the loss
// goes down", but "the architecture weights (alpha) drift toward the
// candidate operation that genuinely fits this problem better" -- when run
// through the *official* onnxruntime-web package (no custom WASM build, no
// onnxruntime.training). Reuses ../federated_lora/step_graph_runner.mjs's
// StepGraphSession/parseManifest/loadInitialState directly: a DARTS
// supernet's manifest is byte-for-byte the same shape a discrete NAS
// candidate's is (see that generator's own docstring on why), it just
// happens to have `alpha` as one more `weight` entry alongside the two
// linear layers.
//
// Skips (does not fail) if onnxruntime-web is not installed here, matching
// ../federated_lora/step_graph_runner.test.mjs's own convention -- `npm
// install` in this directory to run it for real. generate_darts_supernet_
// step_graph.py needs onnxsim's own C++ extension built (see the repo's
// CLAUDE.md) -- this test does not attempt to build it, only to run it.
//
// **The tuning this test's hyperparameters needed, found empirically, not
// assumed.** The first version of this test trained against the same
// dim-normalized (`/ sqrt(dim)`), unit-scale synthetic batch the discrete
// search's own test uses, on the belief that a purely linear target should
// make DARTS prefer the parameter-free Identity branch over the two
// Sigmoid branches (an Identity-only path is an unconstrained linear map,
// an exact fit modulo noise; either Sigmoid path adds unwanted
// nonlinearity). Training it for real (via onnx.reference.ReferenceEvaluator
// against the generator's own output, before this test existed) showed
// that hypothesis does NOT hold at that scale: every candidate can shrink
// its own pre-activation into Sigmoid's near-linear regime around zero
// (`sigmoid(z) ~= 0.5 + z/4` for small `z`) and approximate a linear map
// just as well as Identity does there, so alpha barely moved off uniform.
// Scaling the input up (`inputScale: 5`) and using the raw (non-dim-
// normalized) target widens the range z1 needs to cover to fit the data;
// Sigmoid's bounded [0, 1] range can no longer track that no matter how it
// is scaled downstream, while Identity's unbounded range still can -- and
// only at that scale does alpha's own preference for Identity actually
// emerge and grow over training, verified against the reference evaluator
// before being pinned here.
//
// Usage:
//   npm install && npm test -- darts_supernet.test.mjs
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { StepGraphSession, adamBiasCorrections, loadInitialState, parseManifest } from "../federated_lora/step_graph_runner.mjs";
import { makeSyntheticBatch, softmax } from "./search.mjs";

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

test(
  "DARTS supernet's architecture weights drift toward the Identity candidate on a wide-dynamic-range linear target",
  { skip: ort === null && "onnxruntime-web not installed -- npm install to run this test" },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "onnx-finetune-darts-supernet-"));
    const batchSize = 32, dim = 16, hidden = 16, out = 8;
    const stepPath = join(dir, "step.onnx");
    runPython("generate_darts_supernet_step_graph.py", [
      "-o", stepPath, "--batch-size", String(batchSize), "--dim", String(dim), "--hidden", String(hidden), "--out", String(out),
    ]);

    const manifest = parseManifest(readFileSync(`${stepPath}.manifest.txt`, "utf8"));
    // The supernet's manifest carries "alpha" as an ordinary weight entry,
    // same as L1_w/L1_b/L2_w/L2_b -- nothing about the runner needs to know
    // it is architecture-search-specific.
    assert.ok(manifest.weights.some((w) => w.name === "alpha"), "expected an 'alpha' weight entry");
    assert.deepEqual(manifest.weights.find((w) => w.name === "alpha").shape, [3]);

    // A Node Buffer's own .buffer is the underlying (possibly pooled,
    // larger) ArrayBuffer, not a slice sized to this file -- .slice() by
    // this Buffer's own byteOffset/byteLength is what actually recovers
    // just the bytes this file contains (the same fix
    // ../federated_lora/step_graph_runner.test.mjs's own trainOneClient,
    // and ./search.test.mjs's own loadCandidates, already apply to the
    // same readFileSync(...).buffer pattern -- caught here the same way:
    // an all-garbage initial state reads back as an immediate NaN loss).
    const initialStateBytes = readFileSync(`${stepPath}.initial_state.bin`);
    let state = loadInitialState(
      manifest,
      initialStateBytes.buffer.slice(initialStateBytes.byteOffset, initialStateBytes.byteOffset + initialStateBytes.byteLength),
    );
    const initialAlpha = softmax(Array.from(state.alpha));

    const session = await StepGraphSession.create(
      ort,
      new Uint8Array(readFileSync(stepPath)),
      // wasm, not webgpu: WebGPU is unreachable under plain Node (no
      // navigator.gpu) -- see ../federated_lora/step_graph_runner.test.mjs's
      // own comment on why this still exercises the executionProviders
      // plumbing WebGPU shares, and this module's own docstring on why every
      // op here (Exp/ReduceSum/Div included, hand-building softmax rather
      // than using the fused op) stays inside qat_graph.EP_FRIENDLY_OPS.
      { executionProviders: ["wasm"] },
    );

    const { batchInput, target } = makeSyntheticBatch(batchSize, dim, out, 0, {
      inputScale: 5,
      normalizeByDim: false,
    });

    const lr = 0.05;
    const numSteps = 1500;
    const losses = [];
    for (let t = 0; t < numSteps; ++t) {
      const { loss, state: nextState } = await session.step(manifest, state, batchInput, target, lr, t);
      assert.ok(Number.isFinite(loss), `loss ${loss} is not finite at step ${t}`);
      losses.push(loss);
      state = nextState;
    }
    await session.release?.();

    assert.ok(
      losses[losses.length - 1] < 0.05 * losses[0],
      `loss should drop sharply: ${losses[0]} -> ${losses[losses.length - 1]}`,
    );

    const finalAlpha = softmax(Array.from(state.alpha));
    const [identityWeight, sigmoidWeight, doubleSigmoidWeight] = finalAlpha;

    assert.ok(
      identityWeight > sigmoidWeight && identityWeight > doubleSigmoidWeight,
      `Identity's mixture weight should be the largest of the three: ${JSON.stringify(finalAlpha)}`,
    );
    assert.ok(
      identityWeight > initialAlpha[0] + 0.05,
      `Identity's mixture weight should have grown meaningfully from its uniform start ` +
        `(${initialAlpha[0]} -> ${identityWeight})`,
    );
  },
);

test("adamBiasCorrections is reused correctly across a longer run (sanity check for the 1500-step test above)", () => {
  const t0 = adamBiasCorrections(0);
  const t1499 = adamBiasCorrections(1499);
  // Both bias corrections decay toward 1 as t grows; just confirm they
  // move in the right direction rather than staying pinned at their t=0
  // values, which would silently turn every later Adam step into the same
  // (wrong) update as the first one.
  assert.ok(t1499.mCorrection < t0.mCorrection);
  assert.ok(t1499.vCorrection < t0.vCorrection);
});
