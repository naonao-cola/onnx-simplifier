// Unit test for ort_log_capture.mjs — the helper that mirrors onnxruntime-web's
// own console diagnostics and unhandled WebNN/ORT promise rejections into the
// inference panel's UI log. The DOM-touching pieces take an injected console and
// event target, so the whole thing runs under Node with no browser.
//
// Usage:
//   node test/ort_log_capture.test.mjs

import assert from "node:assert/strict";
import {
  isOrtDiagnostic,
  valueToText,
  formatLogArgs,
  captureOrtDiagnostics,
} from "../ort_log_capture.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

// A console-shaped stub recording each warn/error call's arguments.
function fakeConsole() {
  const calls = { warn: [], error: [] };
  return {
    calls,
    warn: (...a) => calls.warn.push(a),
    error: (...a) => calls.error.push(a),
  };
}

// A minimal EventTarget-shaped stub that can dispatch to registered listeners
// and report how many are still registered (so restore() can be checked).
function fakeTarget() {
  const listeners = {};
  return {
    addEventListener(type, fn) {
      (listeners[type] ||= []).push(fn);
    },
    removeEventListener(type, fn) {
      listeners[type] = (listeners[type] || []).filter((f) => f !== fn);
    },
    dispatch(type, event) {
      (listeners[type] || []).slice().forEach((f) => f(event));
    },
    count(type) {
      return (listeners[type] || []).length;
    },
  };
}

check("isOrtDiagnostic matches onnxruntime/webnn text only", () => {
  assert.equal(isOrtDiagnostic("WebNN backend does not support data type: int64"), true);
  assert.equal(
    isOrtDiagnostic("[W:onnxruntime:, session_state.cc] VerifyEachNodeIsAssignedToAnEp"),
    true,
  );
  assert.equal(isOrtDiagnostic("some unrelated page log"), false);
  assert.equal(isOrtDiagnostic(""), false);
  assert.equal(isOrtDiagnostic(undefined), false);
  assert.equal(isOrtDiagnostic(42), false);
});

check("valueToText normalizes errors, strings, and objects", () => {
  assert.equal(valueToText(new Error("boom")), "boom");
  assert.equal(valueToText("plain"), "plain");
  assert.equal(valueToText({ a: 1 }), '{"a":1}');
  // Circular objects can't be JSON-encoded; fall back to String() without throwing.
  const circular = {};
  circular.self = circular;
  assert.equal(valueToText(circular), "[object Object]");
});

check("formatLogArgs joins mixed args into one matchable string", () => {
  assert.equal(
    formatLogArgs(["WebNN backend does not support data type:", new Error("int64")]),
    "WebNN backend does not support data type: int64",
  );
  assert.equal(formatLogArgs([]), "");
});

check("captureOrtDiagnostics mirrors ORT console diagnostics, not others", () => {
  const con = fakeConsole();
  const target = fakeTarget();
  const logged = [];
  const restore = captureOrtDiagnostics({ log: (m) => logged.push(m), con, target });

  // An ORT warning is forwarded to the log AND still reaches the real console.
  con.warn("[W:onnxruntime] VerifyEachNodeIsAssignedToAnEp: nodes unassigned");
  // An unrelated console.error is left alone (devtools only).
  con.error("some other library failed");

  assert.equal(con.calls.warn.length, 1); // original still called
  assert.equal(con.calls.error.length, 1);
  assert.equal(logged.length, 1);
  assert.match(logged[0], /^onnxruntime-web warn: /);
  assert.match(logged[0], /VerifyEachNodeIsAssignedToAnEp/);

  restore();
});

check("captureOrtDiagnostics mirrors unhandled WebNN rejections", () => {
  const con = fakeConsole();
  const target = fakeTarget();
  const logged = [];
  const restore = captureOrtDiagnostics({ log: (m) => logged.push(m), con, target });

  assert.equal(target.count("unhandledrejection"), 1);

  // The real-world case: an uncaught promise rejection from ORT's WebNN backend.
  target.dispatch("unhandledrejection", {
    reason: new Error("WebNN backend does not support data type: int64"),
  });
  // A non-ORT rejection is ignored.
  target.dispatch("unhandledrejection", { reason: new Error("unrelated") });

  assert.equal(logged.length, 1);
  assert.match(logged[0], /^onnxruntime-web unhandled rejection: /);
  assert.match(logged[0], /int64/);

  restore();
});

// Node has no global GPUDevice; a fakeTarget() stub (already used for
// `window` above) has the same addEventListener/removeEventListener shape, so
// it doubles as a fake device here.
async function checkAsync(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

await checkAsync(
  "captureOrtDiagnostics mirrors an uncaptured WebGPU device error",
  async () => {
    const target = fakeTarget();
    const device = fakeTarget();
    const ort = { env: { webgpu: { device: Promise.resolve(device) } } };
    const logged = [];
    const restore = captureOrtDiagnostics({
      log: (m) => logged.push(m),
      con: fakeConsole(),
      target,
      ort,
    });

    // The device promise resolves asynchronously; give it a tick to attach.
    await Promise.resolve();
    await Promise.resolve();

    assert.equal(device.count("uncapturederror"), 1);
    device.dispatch("uncapturederror", {
      error: { message: "storage buffers (9) ... exceeds ... limit (8)" },
    });
    assert.equal(logged.length, 1);
    assert.match(logged[0], /^WebGPU uncaptured error: /);
    assert.match(logged[0], /storage buffers/);

    restore();
    assert.equal(device.count("uncapturederror"), 0);
  },
);

await checkAsync(
  "captureOrtDiagnostics without `ort` never touches a device",
  async () => {
    const target = fakeTarget();
    const logged = [];
    const restore = captureOrtDiagnostics({
      log: (m) => logged.push(m),
      con: fakeConsole(),
      target,
    });
    await Promise.resolve();
    restore(); // must not throw with no device ever attached
    assert.equal(logged.length, 0);
  },
);

check("restore() removes console wrappers and the rejection listener", () => {
  const con = fakeConsole();
  const target = fakeTarget();
  const origWarn = con.warn;
  const origError = con.error;
  const logged = [];
  const restore = captureOrtDiagnostics({ log: (m) => logged.push(m), con, target });

  assert.notEqual(con.warn, origWarn); // wrapped while active
  restore();

  assert.equal(con.warn, origWarn); // restored
  assert.equal(con.error, origError);
  assert.equal(target.count("unhandledrejection"), 0); // listener removed

  // After restore, ORT diagnostics are no longer mirrored.
  con.warn("[W:onnxruntime] still talking");
  target.dispatch("unhandledrejection", { reason: new Error("webnn later") });
  assert.equal(logged.length, 0);
});

console.log(`\nort_log_capture: ${passed} checks passed`);
