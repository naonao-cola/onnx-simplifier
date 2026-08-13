// Turn a model's raw output tensor into something a human can read, for the
// inference panel's sample-data input/output preview. Pure — no DOM/network —
// so the interesting logic (softmax, top-K, the "does this look like an
// ImageNet classifier" heuristic) is unit-testable without a browser.
//
// This is inherently best-effort: an ONNX graph's output carries no label
// metadata, so there's no reliable way to know what a given output tensor
// *means*. The one heuristic used here — a length-1000 last dimension likely
// being a standard ImageNet-1k classifier — covers most of the curated vision
// models (timm/torchvision-trained) but will mislabel a model trained on a
// different 1000-class task. Anything else falls back to a plain numeric
// preview (shape + first few values) rather than guessing further.

import { IMAGENET_CLASSES } from "./imagenet_classes.mjs";

// True when `dims`' last axis is exactly 1000 and every other axis is 1 (a
// single-sample 1000-way classifier output, e.g. [1, 1000] or [1000]).
export function isImagenetLogits(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return false;
  const last = dims[dims.length - 1];
  if (last !== 1000) return false;
  return dims.slice(0, -1).every((d) => d === 1);
}

// Numerically-stable softmax over a flat array-like of numbers. Returns a
// plain Array of probabilities summing to ~1.
export function softmax(values) {
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  const exps = Array.from(values, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return sum > 0 ? exps.map((e) => e / sum) : exps.map(() => 0);
}

// Top-K { index, prob } pairs from a probability array, highest first.
export function topK(probs, k = 5) {
  return Array.from(probs, (prob, index) => ({ index, prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, Math.max(0, k));
}

// Summarize a raw model output for display: { kind: "imagenet", top: [{label,
// prob}, ...] } when it looks like a standard 1000-way ImageNet classifier, or
// { kind: "raw", preview: [numbers...], total } otherwise (the first few
// values, for a quick sanity glance rather than a full dump).
export function summarizeOutput(values, dims, k = 5) {
  if (isImagenetLogits(dims)) {
    const probs = softmax(values);
    const top = topK(probs, k).map(({ index, prob }) => ({
      label: IMAGENET_CLASSES[index] || `class ${index}`,
      prob,
    }));
    return { kind: "imagenet", top };
  }
  const total = values.length;
  const preview = Array.from(values.slice(0, 8), (v) => Number(v));
  return { kind: "raw", preview, total };
}
