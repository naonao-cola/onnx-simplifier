// Unit test for the pure issue-report URL builder behind the converter page's
// "Report an issue" button. No DOM or network needed.
//
// Usage:
//   node test/issue_report.test.mjs

import assert from "node:assert/strict";
import { buildIssueUrl, DEFAULT_MAX_URL_LENGTH } from "../issue_report.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

const REPO = "https://github.com/onnxsim/onnxsim";
const TITLE = "[BUG] WASM converter: ";

check("includes the page, user agent, and run info", () => {
  const url = buildIssueUrl({
    repo: REPO,
    title: TITLE,
    pageUrl: "https://example.com/convert",
    userAgent: "TestAgent/1.0",
    runInfo: {
      optimizer: "simplify",
      file_name: "m.onnx",
      constant_fold: true,
      shape_inference: true,
      tensor_size_threshold: 100,
      target_opset_version: 0,
      passes: ["fuse_bn_into_conv"],
    },
    log: "hello world",
  });
  assert.ok(url.startsWith(REPO + "/issues/new?title="));
  const body = decodeURIComponent(url.split("&body=")[1]);
  assert.ok(body.includes("https://example.com/convert"));
  assert.ok(body.includes("TestAgent/1.0"));
  assert.ok(body.includes("Optimizer: simplify"));
  assert.ok(body.includes("Skipped passes: fuse_bn_into_conv"));
  assert.ok(body.includes("hello world"));
});

check("notes when no conversion has run", () => {
  const url = buildIssueUrl({ repo: REPO, title: TITLE, runInfo: null, log: "" });
  const body = decodeURIComponent(url.split("&body=")[1]);
  assert.ok(body.includes("no conversion has been run yet"));
  assert.ok(body.includes("(no console output)"));
});

check("keeps a short URL untrimmed", () => {
  const url = buildIssueUrl({ repo: REPO, title: TITLE, log: "a short log" });
  assert.ok(url.length < DEFAULT_MAX_URL_LENGTH);
  const body = decodeURIComponent(url.split("&body=")[1]);
  assert.ok(body.includes("a short log"));
  assert.ok(!body.includes("truncated"));
});

check("trims a huge log so the URL stays under the cap", () => {
  // A log far larger than any URL budget, with a unique marker at the very end.
  const big = Array.from({ length: 5000 }, (_, i) => `line ${i} of noisy log output`).join("\n");
  const log = big + "\nFINAL_ERROR_MARKER: boom";
  const url = buildIssueUrl({ repo: REPO, title: TITLE, log });
  assert.ok(
    url.length <= DEFAULT_MAX_URL_LENGTH,
    `URL length ${url.length} should be <= ${DEFAULT_MAX_URL_LENGTH}`,
  );
  const body = decodeURIComponent(url.split("&body=")[1]);
  // The error-bearing tail is preserved; a truncation note is added.
  assert.ok(body.includes("FINAL_ERROR_MARKER: boom"));
  assert.ok(body.includes("truncated"));
  // The environment block is never trimmed away.
  assert.ok(body.includes("**Environment (WASM converter)**"));
});

check("respects a custom max length", () => {
  const log = Array.from({ length: 2000 }, (_, i) => `x${i}`).join("\n");
  const url = buildIssueUrl({ repo: REPO, title: TITLE, log, maxUrlLength: 2000 });
  assert.ok(url.length <= 2000, `URL length ${url.length} should be <= 2000`);
});

console.log(`PASS: ${passed} checks`);
