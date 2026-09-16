// Parser test for the "Live Cardputer output (Web Serial)" panel.
//
// Verifies ingestLine()/renderLatest() against real lines captured from an
// actual M5Stack Cardputer running onnx-cardputer-flash's runtime (see
// tools/onnx-cardputer-flash/firmware/runtime/src/main.cpp's
// RunAndReportInference()/printOutputValues()) -- not synthesized from
// reading that file's format strings, so a mismatch here would have caught
// the same kind of drift a hand-written fixture could paper over.
//
//   node test/cardputer_monitor_ui.test.mjs

import assert from "node:assert/strict";
import { ingestLine, renderLatest } from "../cardputer_monitor_ui.mjs";

function freshState() {
  return { status: null, latencyMs: 0, outputs: new Map() };
}

// A real capture: boot lines (unparsed, but must not throw or false-positive
// as a status/output line), then one full inference cycle.
const REAL_CAPTURE = [
  "ESP-ROM:esp32s3-20210327",
  "onnx-cardputer-flash runtime",
  "mapping model partition...",
  "model loaded ok.",
  "inputs: 1  outputs: 1",
  "  in[0]: 1x40x1x49 type=9",
  "mic ready -- feeding real audio into the model below",
  "test inference: OK (47 ms)",
  "  out[0] raw: -64,-65,-65,-63",
  "  out[0] dequant: 0.2500,0.2461,0.2461,0.2539",
];

{
  const state = freshState();
  let resultLines = 0;
  for (const line of REAL_CAPTURE) {
    if (ingestLine(state, line)) resultLines++;
  }
  // Only the "test inference:" and two "out[0] ..." lines are result lines;
  // the boot/status lines above them must not match either pattern.
  assert.equal(resultLines, 3, "exactly 3 lines should be recognized as result lines");
  assert.equal(state.status, "OK");
  assert.equal(state.latencyMs, 47);
  assert.equal(state.outputs.size, 1);
  assert.deepEqual(state.outputs.get(0).raw, [-64, -65, -65, -63]);
  assert.deepEqual(state.outputs.get(0).dequant, [0.25, 0.2461, 0.2461, 0.2539]);

  const html = renderLatest(state);
  assert.match(html, /OK/);
  assert.match(html, /47 ms/);
  assert.match(html, /-64, -65, -65, -63/);
  console.log("real-capture parse: ok");
}

// A FAILED cycle (e.g. finding #4's Transpose/uint8 error) prints a status
// line but no out[] lines -- renderLatest() must not crash on an empty
// outputs map.
{
  const state = freshState();
  assert.equal(ingestLine(state, "test inference: FAILED (12 ms)"), true);
  assert.equal(state.status, "FAILED");
  assert.equal(state.outputs.size, 0);
  const html = renderLatest(state);
  assert.match(html, /FAILED/);
  assert.match(html, /12 ms/);
  console.log("FAILED-status parse: ok");
}

// A float32 model's "out[i] values: ..." line (main.cpp's other branch,
// distinct from the int8/uint8 "raw"/"dequant" pair) is read into the same
// bar-rendering path as "dequant".
{
  const state = freshState();
  ingestLine(state, "test inference: OK (9 ms)");
  ingestLine(state, "  out[0] values: 0.1234,0.9000");
  assert.deepEqual(state.outputs.get(0).values, [0.1234, 0.9]);
  const html = renderLatest(state);
  assert.match(html, /0\.1234/);
  console.log("float32 values parse: ok");
}

// A long output line ends with printOutputValues()'s literal "..." --
// Number("...") is NaN and must be dropped, not rendered as a NaN bar.
{
  const state = freshState();
  ingestLine(state, "test inference: OK (5 ms)");
  ingestLine(state, "  out[0] raw: 1,2,3,...");
  assert.deepEqual(state.outputs.get(0).raw, [1, 2, 3]);
  console.log("truncated-output '...' handling: ok");
}

// Before any status line arrives, renderLatest() must render a placeholder,
// not throw or reference a null status.
{
  const html = renderLatest(freshState());
  assert.match(html, /no inference reported yet/);
  console.log("pre-connect placeholder: ok");
}

console.log("cardputer_monitor_ui.test.mjs: all checks passed");
