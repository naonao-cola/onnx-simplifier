// Unit tests for the pure parts of sample_inputs.mjs: the image/text shape
// classifiers and the pixel-normalization math. The DOM-coupled pieces
// (createImageBitmap/canvas decode, the actual network fetch) aren't exercised
// here — consistent with inference_browser.mjs itself having no DOM-level Node
// test — and are covered by manual testing (see test/README.md).
//
// Usage:
//   node test/sample_inputs.test.mjs

import assert from "node:assert/strict";
import { isImageInput, isTextInput, normalizePixels } from "../sample_inputs.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("isImageInput accepts a float32 NCHW RGB input", () => {
  assert.equal(isImageInput({ type: "float32", name: "input" }, [1, 3, 224, 224]), true);
});

check("isImageInput accepts a grayscale (1-channel) input", () => {
  assert.equal(isImageInput({ type: "float32", name: "input" }, [1, 1, 28, 28]), true);
});

check("isImageInput rejects non-float32, wrong rank, or a non-1/3 channel dim", () => {
  assert.equal(isImageInput({ type: "int64", name: "input" }, [1, 3, 224, 224]), false);
  assert.equal(isImageInput({ type: "float32", name: "input" }, [1, 3, 224]), false);
  assert.equal(isImageInput({ type: "float32", name: "input" }, [1, 4, 224, 224]), false);
  assert.equal(isImageInput({ type: "float32", name: "input" }, [1, 3, 0, 224]), false);
});

check("isTextInput accepts int64 rank-2 inputs named like tokenizer outputs", () => {
  for (const name of ["input_ids", "attention_mask", "token_type_ids", "decoder_input_ids"]) {
    assert.equal(isTextInput({ type: "int64", name }, [1, 128]), true, name);
  }
});

check("isTextInput rejects wrong dtype, rank, or an unrecognized name", () => {
  assert.equal(isTextInput({ type: "float32", name: "input_ids" }, [1, 128]), false);
  assert.equal(isTextInput({ type: "int64", name: "input_ids" }, [1, 128, 1]), false);
  assert.equal(isTextInput({ type: "int64", name: "pixel_values" }, [1, 128]), false);
});

check("normalizePixels maps mid-gray RGB to ~0 with ImageNet mean/std", () => {
  const w = 2, h = 1, c = 3;
  const rgba = new Uint8ClampedArray(w * h * 4);
  // Two pixels at the ImageNet mean (scaled to 0..255) should normalize to ~0.
  const meanByte = [0.485, 0.456, 0.406].map((m) => Math.round(m * 255));
  for (let p = 0; p < w; p++) {
    rgba[p * 4 + 0] = meanByte[0];
    rgba[p * 4 + 1] = meanByte[1];
    rgba[p * 4 + 2] = meanByte[2];
    rgba[p * 4 + 3] = 255;
  }
  const out = normalizePixels(rgba, c, h, w);
  assert.equal(out.length, c * h * w);
  for (const v of out) assert.ok(Math.abs(v) < 0.02, `expected ~0, got ${v}`);
});

check("normalizePixels lays out NCHW (channel-major)", () => {
  const w = 2, h = 2, c = 3;
  const rgba = new Uint8ClampedArray(w * h * 4).fill(0);
  rgba[3] = 255; // alpha channels only; RGB stays 0
  const out = normalizePixels(rgba, c, h, w);
  // All-zero RGB -> every channel plane should hold the same normalized value
  // (mean/std differ per channel, but are constant within a channel here).
  const plane = h * w;
  for (let ch = 0; ch < c; ch++) {
    const first = out[ch * plane];
    for (let i = 0; i < plane; i++) {
      assert.equal(out[ch * plane + i], first);
    }
  }
});

check("normalizePixels averages RGB into one channel for grayscale (c=1)", () => {
  const w = 1, h = 1, c = 1;
  const rgba = new Uint8ClampedArray([30, 60, 90, 255]);
  const out = normalizePixels(rgba, c, h, w);
  const expected = (30 + 60 + 90) / 3 / 255 / 0.5 - 0.5 / 0.5;
  assert.ok(Math.abs(out[0] - expected) < 1e-6);
});

console.log(`PASS: ${passed} checks`);
