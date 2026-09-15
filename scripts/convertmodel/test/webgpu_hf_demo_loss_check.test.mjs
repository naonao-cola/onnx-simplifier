// Regression test for webgpu_hf_demo.test.mjs's "loss decreased
// meaningfully over the run" check: replays the exact step_qat_hf_demo.onnx
// graph (same baked w1/b1/w2/b2 initial weights, same lr=0.01/40-step Adam
// schedule) against webgpu_hf_demo_adversarial_x.json -- an input found
// (see make_webgpu_hf_demo_regression_fixture.py's own docstring) whose
// *initial*, untrained forward pass already lands within float32 precision
// of the target (0). That reproduces exactly what a live CI run hit: with
// losses[0] a tiny, near-coincidental outlier, Adam's bias-corrected first
// step (roughly lr * sign(gradient) regardless of how small the gradient
// already is) necessarily jumps the loss up by orders of magnitude in
// *relative* terms on step 1, even though training proceeds completely
// normally afterward -- which a naive losses[-1] < 0.8 * losses[0] check
// cannot tell apart from an actually-broken run.
//
// Runs entirely under plain Node against onnxruntime-web's wasm backend --
// no browser, no live network, no GPU -- so it's part of `npm run test:all`
// and catches a regression back to comparing raw endpoints without needing
// CI to get unlucky with a fresh live photo again.
//
// Usage: node test/webgpu_hf_demo_loss_check.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { runStepLoop } from "../webgpu_hf_demo.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = join(HERE, "..", "node_modules", "onnxruntime-web", "dist") + "/";

function average(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

async function main() {
  // Plain "onnxruntime-web" resolves to the Node bundle (cpu/wasm only) --
  // exactly what's needed here, same as step_graph_ep.test.mjs's "wasm"
  // target.
  const ortMod = await import("onnxruntime-web");
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = DIST;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;

  const manifest = JSON.parse(readFileSync(join(HERE, "step_qat_hf_demo.json"), "utf8"));
  const fixture = JSON.parse(readFileSync(join(HERE, "webgpu_hf_demo_adversarial_x.json"), "utf8"));
  const modelBytes = readFileSync(join(HERE, "step_qat_hf_demo.onnx"));

  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "disabled",
  });
  const constants = {
    x: new ort.Tensor("float32", Float32Array.from(fixture.x), [1, manifest.inputDim]),
    teacher: new ort.Tensor("float32", new Float32Array([0]), [1, 1]),
  };
  const { losses } = await runStepLoop(ort, session, manifest, constants);
  await session.release?.();

  console.log(
    `adversarial x: losses[0]=${losses[0].toExponential(3)} ` +
      `losses[-1]=${losses[losses.length - 1].toExponential(3)}`,
  );

  let passed = 0;
  function check(name, fn) {
    fn();
    passed += 1;
    console.log("  ok -", name);
  }

  check("losses[0] reproduces the near-zero outlier this fixture is for", () => {
    // Not exactly fixture.initialOutput**2 -- the graph's own float32
    // arithmetic, not the fixture generator's float64 forward pass -- but
    // should still land far below a normal run's initial loss (~1e-2 to
    // 1e-1 across ordinary photos; see webgpu_hf_demo.test.mjs's own
    // reproductions).
    assert.ok(losses[0] < 1e-6, `expected a near-zero initial loss, got ${losses[0]}`);
  });

  check("every loss is finite", () => {
    for (const loss of losses) assert.ok(Number.isFinite(loss), `loss ${loss} is not finite`);
  });

  check("the raw first/last endpoint comparison FAILS for this input", () => {
    // Confirms this fixture still demonstrates the bug -- if this ever
    // starts passing, the reproduction has stopped reproducing anything
    // (e.g. the graph or lr schedule changed) and needs a fresh look, not a
    // deleted assertion.
    const first = losses[0];
    const last = losses[losses.length - 1];
    assert.ok(
      !(last < 0.8 * first),
      `expected the naive endpoint check to fail here (it would have caught ` +
        `nothing) but got ${first} -> ${last}`,
    );
  });

  check("the windowed-average comparison PASSES for this input", () => {
    const WINDOW = Math.min(5, Math.floor(losses.length / 2));
    const early = average(losses.slice(0, WINDOW));
    const late = average(losses.slice(-WINDOW));
    assert.ok(
      late < 0.8 * early,
      `windowed check should pass for this adversarial x: early ${early} -> late ${late}`,
    );
  });

  console.log(`\nwebgpu hf demo loss check: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
