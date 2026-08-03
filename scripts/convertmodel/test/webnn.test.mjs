// Unit test for the pure WebNN helpers behind the converter page's inference
// panel EP picker (webnn.mjs). No DOM, no onnxruntime-web, no real navigator —
// detectWebnn() takes an injected navigator so the probe is testable.
//
// Usage:
//   node test/webnn.test.mjs

import assert from "node:assert/strict";
import {
  WEBNN_DEVICE_TYPES,
  webnnProvider,
  providerLabel,
  providersForEp,
  isWebnnEp,
  detectWebnn,
  formatWebnnStatus,
} from "../webnn.mjs";

let passed = 0;
// Runs fn (sync or async) and logs a tick. Callers `await` it so the async
// probes finish in order before the final count.
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// webnnProvider builds the ORT options object with a powerPreference only for
// the accelerator devices.
await check("webnnProvider shapes the ORT execution-provider entry", () => {
  assert.deepEqual(webnnProvider("gpu"), {
    name: "webnn",
    deviceType: "gpu",
    powerPreference: "default",
  });
  assert.deepEqual(webnnProvider("npu"), {
    name: "webnn",
    deviceType: "npu",
    powerPreference: "default",
  });
  assert.deepEqual(webnnProvider("cpu"), { name: "webnn", deviceType: "cpu" });
  assert.throws(() => webnnProvider("tpu"), /unknown WebNN device type/);
});

// providerLabel keeps existing string EPs intact and gives WebNN objects a
// readable name (so inference-core's log/returned ep stays a string).
await check("providerLabel renders strings and WebNN objects", () => {
  assert.equal(providerLabel("wasm"), "wasm");
  assert.equal(providerLabel("webgpu"), "webgpu");
  assert.equal(providerLabel(webnnProvider("gpu")), "webnn:gpu");
  assert.equal(providerLabel({ name: "webnn" }), "webnn");
  assert.equal(providerLabel({ name: "cuda" }), "cuda");
});

// Every WebNN/WebGPU choice ends in a wasm fallback; needWebnn flags the choices
// that require the WebNN-capable bundle.
await check("providersForEp maps dropdown values to provider lists", () => {
  assert.deepEqual(providersForEp("wasm"), {
    providers: ["wasm"],
    needWebnn: false,
  });
  assert.deepEqual(providersForEp("webgpu"), {
    providers: ["webgpu", "wasm"],
    needWebnn: false,
  });

  const gpu = providersForEp("webnn-gpu");
  assert.equal(gpu.needWebnn, true);
  assert.deepEqual(gpu.providers, [webnnProvider("gpu"), "wasm"]);

  const npu = providersForEp("webnn-npu");
  assert.equal(npu.needWebnn, true);
  assert.deepEqual(npu.providers, [webnnProvider("npu"), "wasm"]);

  const cpu = providersForEp("webnn-cpu");
  assert.equal(cpu.needWebnn, true);
  assert.deepEqual(cpu.providers, [webnnProvider("cpu"), "wasm"]);

  // Unknown values degrade to plain wasm.
  assert.deepEqual(providersForEp("nonsense"), {
    providers: ["wasm"],
    needWebnn: false,
  });
});

await check("isWebnnEp recognizes only the WebNN dropdown values", () => {
  assert.equal(isWebnnEp("webnn-gpu"), true);
  assert.equal(isWebnnEp("webnn-cpu"), true);
  assert.equal(isWebnnEp("webgpu"), false);
  assert.equal(isWebnnEp("wasm"), false);
  assert.equal(isWebnnEp(undefined), false);
});

// A console-shaped stub that records the arguments of each error() call, so the
// tests can assert detectWebnn surfaces the underlying WebNN failures without
// touching the real console.
function fakeConsole() {
  const errors = [];
  return { errors, error: (...args) => errors.push(args) };
}

// detectWebnn with no navigator.ml reports the API as absent (never throws), and
// with no device probes there is nothing to log.
await check("detectWebnn reports absent API", async () => {
  const con = fakeConsole();
  const report = await detectWebnn({}, con);
  assert.equal(report.apiPresent, false);
  assert.equal(report.supported, false);
  assert.deepEqual(report.errors, {});
  assert.match(report.message, /not available/);
  assert.equal(con.errors.length, 0);
  assert.equal(formatWebnnStatus(report).startsWith("WebNN status: ❌"), true);
});

// A navigator whose ml.createContext succeeds only for some device types: those
// become supported, the rest false, and supported reflects "at least one". The
// failing device types' errors are captured in the report and logged to console.
await check("detectWebnn probes each device type", async () => {
  const calls = [];
  const nav = {
    ml: {
      async createContext({ deviceType }) {
        calls.push(deviceType);
        if (deviceType === "gpu") return { destroy() {} };
        throw new Error(`no ${deviceType}`);
      },
    },
  };
  const con = fakeConsole();
  const report = await detectWebnn(nav, con);
  assert.deepEqual(calls, WEBNN_DEVICE_TYPES); // probed gpu, npu, cpu
  assert.equal(report.apiPresent, true);
  assert.equal(report.supported, true);
  assert.deepEqual(report.devices, { gpu: true, npu: false, cpu: false });
  assert.deepEqual(report.errors, { npu: "no npu", cpu: "no cpu" });
  // Only the two failing device types are logged; the succeeding gpu is not.
  assert.equal(con.errors.length, 2);
  assert.match(con.errors[0][0], /deviceType 'npu'/);
  assert.equal(con.errors[0][1].message, "no npu");
  assert.match(report.message, /gpu/);
  assert.equal(formatWebnnStatus(report).startsWith("WebNN status: ✅"), true);
});

// navigator.ml present but no device usable → apiPresent true, supported false,
// warning glyph, every device's error captured and logged.
await check("detectWebnn handles API present but no usable device", async () => {
  const nav = {
    ml: {
      async createContext() {
        throw new Error("unsupported");
      },
    },
  };
  const con = fakeConsole();
  const report = await detectWebnn(nav, con);
  assert.equal(report.apiPresent, true);
  assert.equal(report.supported, false);
  assert.deepEqual(report.devices, { gpu: false, npu: false, cpu: false });
  assert.deepEqual(report.errors, {
    gpu: "unsupported",
    npu: "unsupported",
    cpu: "unsupported",
  });
  assert.equal(con.errors.length, 3);
  assert.equal(formatWebnnStatus(report).startsWith("WebNN status: ⚠️"), true);
});

// A createContext that rejects with a non-Error value is stringified rather than
// dropped, so the report and console still carry something useful.
await check("detectWebnn stringifies non-Error rejections", async () => {
  const nav = {
    ml: {
      async createContext() {
        throw "boom"; // eslint-disable-line no-throw-literal
      },
    },
  };
  const con = fakeConsole();
  const report = await detectWebnn(nav, con);
  assert.deepEqual(report.errors, { gpu: "boom", npu: "boom", cpu: "boom" });
  assert.equal(con.errors.length, 3);
});

console.log(`\nwebnn: ${passed} checks passed`);
