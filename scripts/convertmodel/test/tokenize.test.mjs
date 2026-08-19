// Unit tests for the pure parts of tokenize.mjs (family resolution + the
// FAMILY_TOKENIZER table itself). `Tokenizer`/network are not exercised here —
// tokenizeForModel() needs a real browser fetch + the CDN module, covered by
// manual testing (see test/README.md).
//
// Usage:
//   node test/tokenize.test.mjs

import assert from "node:assert/strict";
import { FAMILY_TOKENIZER, familyFromFilename } from "../tokenize.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

check("every FAMILY_TOKENIZER entry looks like a valid HF repo id", () => {
  for (const [family, repo] of Object.entries(FAMILY_TOKENIZER)) {
    assert.ok(/^[\w.-]+(\/[\w.-]+)?$/.test(repo), `${family} -> '${repo}' looks malformed`);
  }
});

check("familyFromFilename resolves every known family from its onnxmodelzoo filename", () => {
  // Mirrors the actual filenames in scripts/regression/models.json.
  const cases = {
    "bert_Opset18.onnx": "bert",
    "gpt2_Opset18.onnx": "gpt2",
    "albert_Opset18.onnx": "albert",
    "distilbert_Opset18.onnx": "distilbert",
    "roberta_Opset18.onnx": "roberta",
    "xlmroberta_Opset18.onnx": "xlmroberta",
    "electra_Opset18.onnx": "electra",
    "deberta_Opset17.onnx": "deberta",
    "mobilebert_Opset18.onnx": "mobilebert",
    "squeezebert_Opset18.onnx": "squeezebert",
    "bart_Opset18.onnx": "bart",
    "mvp_Opset18.onnx": "mvp",
    "t5_Opset17.onnx": "t5",
    "longt5_Opset17.onnx": "longt5",
  };
  for (const [filename, expected] of Object.entries(cases)) {
    assert.equal(familyFromFilename(filename), expected, filename);
  }
});

check("familyFromFilename also resolves a bare repo id (org/name)", () => {
  assert.equal(familyFromFilename("onnxmodelzoo/bert_Opset18"), "bert");
});

check("familyFromFilename is case-insensitive and separator-tolerant", () => {
  assert.equal(familyFromFilename("BERT_Opset18.ONNX"), "bert");
  assert.equal(familyFromFilename("bert-Opset18"), "bert");
});

check("distilbert doesn't collide with a shorter/unrelated family", () => {
  // Regression guard: distilbert must resolve to itself, not fall through to
  // some other family via a loose substring match.
  assert.equal(familyFromFilename("distilbert_Opset18.onnx"), "distilbert");
  assert.notEqual(familyFromFilename("distilbert_Opset18.onnx"), "bert");
});

check("familyFromFilename returns null for unrecognized / vision model names", () => {
  for (const name of [null, "", "resnet18d_Opset18.onnx", "yolov3-10.onnx", "mystery_model.onnx"]) {
    assert.equal(familyFromFilename(name), null, String(name));
  }
});

console.log(`PASS: ${passed} checks`);
