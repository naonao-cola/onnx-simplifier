// Unit test for webgpu_kernel_profile.mjs's selectDominantNodes -- the pure,
// GPU-free half of the "profile first, then tune only the latency-dominant
// ops" workflow (see webgpu_kernel_annotations_view.mjs's own docstring).
// Synthetic latency numbers only, no browser/GPU/Pyodide involved -- the
// live-profiling half (real per-node GPU timing) is checked end to end in
// webgpu_kernel_tuner_ui.test.mjs's own runProfileScenario instead.
//
// Usage:
//   node test/webgpu_kernel_profile.test.mjs

import assert from "node:assert/strict";
import { selectDominantNodes } from "../webgpu_kernel_profile.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

function main() {
  check("picks the single node that alone covers the default 80% threshold", () => {
    const results = [
      { nodeName: "a", medianMs: 90, skipped: false },
      { nodeName: "b", medianMs: 5, skipped: false },
      { nodeName: "c", medianMs: 5, skipped: false },
    ];
    assert.deepEqual(selectDominantNodes(results), ["a"]);
  });

  check("keeps adding slowest-first until the cumulative threshold is crossed", () => {
    const results = [
      { nodeName: "a", medianMs: 40, skipped: false },
      { nodeName: "b", medianMs: 35, skipped: false },
      { nodeName: "c", medianMs: 15, skipped: false },
      { nodeName: "d", medianMs: 10, skipped: false },
    ];
    // total = 100; a+b = 75 (< 80%), a+b+c = 90 (>= 80%) -- stop there.
    assert.deepEqual(selectDominantNodes(results), ["a", "b", "c"]);
  });

  check("a lower thresholdFraction selects fewer nodes", () => {
    const results = [
      { nodeName: "a", medianMs: 40, skipped: false },
      { nodeName: "b", medianMs: 35, skipped: false },
      { nodeName: "c", medianMs: 15, skipped: false },
      { nodeName: "d", medianMs: 10, skipped: false },
    ];
    // total = 100; a alone = 40 (< 50%), a+b = 75 (>= 50%) -- stop there.
    assert.deepEqual(selectDominantNodes(results, { thresholdFraction: 0.5 }), ["a", "b"]);
  });

  check("skipped nodes (no kernel in isolation, or a profiling error) are never selected", () => {
    const results = [
      { nodeName: "a", medianMs: 0, skipped: true },
      { nodeName: "b", medianMs: 10, skipped: false },
    ];
    assert.deepEqual(selectDominantNodes(results), ["b"]);
  });

  check("returns an empty list when every node was skipped (nothing timed at all)", () => {
    const results = [
      { nodeName: "a", medianMs: 0, skipped: true },
      { nodeName: "b", medianMs: 0, skipped: true },
    ];
    assert.deepEqual(selectDominantNodes(results), []);
  });

  check("minNodes keeps at least that many even if the first alone crosses the threshold", () => {
    const results = [
      { nodeName: "a", medianMs: 90, skipped: false },
      { nodeName: "b", medianMs: 5, skipped: false },
      { nodeName: "c", medianMs: 5, skipped: false },
    ];
    assert.deepEqual(selectDominantNodes(results, { minNodes: 2 }), ["a", "b"]);
  });

  check("an empty input list returns an empty list", () => {
    assert.deepEqual(selectDominantNodes([]), []);
  });

  console.log(`\nwebgpu kernel profile: ${passed} checks passed`);
}

main();
