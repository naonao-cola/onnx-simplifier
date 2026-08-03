// Unit test for the "Run inference" panel's input-fill helper (input_fill.mjs).
//
// The panel auto-generates dummy inputs for the before/after comparison and for
// single-model runs; `fillInput` decides how the float inputs are valued. This
// checks each selectable mode (random/ones/zeros/arange), the safe zero default
// for integer/bool inputs, random determinism (the panel re-feeds the same
// tensor every iteration and asserts a stable output), and the unknown-mode
// guard.
//
// Pure JS, no onnxruntime — runs anywhere node does.
//
// Usage:
//   node test/input_fill.test.mjs

import assert from "node:assert/strict";
import { fillInput, INPUT_FILL_MODES } from "../input_fill.mjs";

function testModes() {
  // ones fills every float slot with 1.
  assert.deepEqual(
    Array.from(fillInput(new Float32Array(4), "float32", "ones")),
    [1, 1, 1, 1],
  );
  // arange is 0, 1, 2, ... in row-major order.
  assert.deepEqual(
    Array.from(fillInput(new Float32Array(4), "float32", "arange")),
    [0, 1, 2, 3],
  );
  // zeros / zero leave the (already zeroed) buffer untouched.
  for (const mode of ["zeros", "zero"]) {
    assert.deepEqual(
      Array.from(fillInput(new Float32Array(3), "float32", mode)),
      [0, 0, 0],
      `${mode} stays all-zero`,
    );
  }
  // float64 is treated as a float too.
  assert.deepEqual(
    Array.from(fillInput(new Float64Array(2), "float64", "ones")),
    [1, 1],
  );
  console.log("PASS modes: ones / arange / zeros produce the expected values");
}

function testRandom() {
  const a = fillInput(new Float32Array(8), "float32", "random");
  const b = fillInput(new Float32Array(8), "float32", "random");
  // Seeded xorshift32: identical across calls (determinism the panel relies on).
  assert.deepEqual(Array.from(a), Array.from(b), "random is deterministic");
  // Non-trivial: not all zero, and within [-1, 1).
  assert.ok(
    Array.from(a).some((v) => v !== 0),
    "random produces non-zero values",
  );
  assert.ok(
    Array.from(a).every((v) => v >= -1 && v < 1),
    "random values fall in [-1, 1)",
  );
  console.log("PASS random: deterministic, non-zero, in [-1, 1)");
}

function testIntegerAndBoolStayZero() {
  // Integer/bool inputs are always left at zero regardless of the mode, the
  // safe default for index-like tensors (Gather, etc.).
  for (const [type, Ctor] of [
    ["int64", BigInt64Array],
    ["int32", Int32Array],
    ["bool", Uint8Array],
  ]) {
    for (const mode of INPUT_FILL_MODES) {
      const out = fillInput(new Ctor(4), type, mode);
      assert.ok(
        Array.from(out).every((v) => v == 0),
        `${type} stays zero under '${mode}'`,
      );
    }
  }
  console.log("PASS integer/bool: left at zero for every fill mode");
}

function testUnknownModeThrows() {
  assert.throws(
    () => fillInput(new Float32Array(2), "float32", "bogus"),
    /unknown input fill/,
    "an unknown mode is rejected",
  );
  console.log("PASS guard: unknown fill mode throws");
}

function main() {
  assert.deepEqual(INPUT_FILL_MODES, ["random", "ones", "zeros", "arange"]);
  testModes();
  testRandom();
  testIntegerAndBoolStayZero();
  testUnknownModeThrows();
  console.log("PASS input_fill: all checks");
}

main();
