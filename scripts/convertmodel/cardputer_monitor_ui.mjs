// Browser glue for the "Live Cardputer output (Web Serial)" panel.
//
// Reads whatever tools/onnx-cardputer-flash/firmware/runtime/src/main.cpp's
// RunAndReportInference()/printOutputValues() already prints over serial --
// this panel doesn't run inference itself, flash anything, or talk to
// onnxruntime-web; it's a live monitor for a physical board's own output,
// rendered in the same page as this tool's other model-inspection panels so
// a device result can sit next to them instead of only living in a separate
// serial terminal.
//
// Loaded as its own top-level module (a <script type="module"> tag in
// index.html), not imported by anything else.

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

const STATUS_LINE = /^test inference: (OK|FAILED) \((\d+) ms\)$/;
const OUTPUT_LINE = /^ {2}out\[(\d+)\] (raw|dequant|values): (.+)$/;

// Parses one board-printed line, updating `state` in place. Returns true if
// this line is part of an inference result worth re-rendering for (the
// caller re-renders the "latest result" panel then; boot/status lines that
// don't match either pattern still go to the raw log, just unparsed).
export function ingestLine(state, line) {
  const statusMatch = line.match(STATUS_LINE);
  if (statusMatch) {
    state.status = statusMatch[1];
    state.latencyMs = Number(statusMatch[2]);
    state.outputs = new Map();
    return true;
  }
  const outputMatch = line.match(OUTPUT_LINE);
  if (outputMatch) {
    const [, indexStr, kind, valuesStr] = outputMatch;
    const index = Number(indexStr);
    // A long output's printOutputValues() caps at 16 elements and appends a
    // literal "...", which Number() turns into NaN -- drop it rather than
    // render a "NaN" bar.
    const values = valuesStr
      .split(",")
      .map((v) => Number(v))
      .filter((v) => !Number.isNaN(v));
    if (!state.outputs.has(index)) state.outputs.set(index, {});
    state.outputs.get(index)[kind] = values;
    return true;
  }
  return false;
}

// Renders `state` (as ingestLine leaves it) into the "latest result" panel's
// innerHTML. Quantized (int8/uint8) tensors show both their raw and
// dequantized values with a small bar per dequantized value (assumes a
// roughly 0..1 range, true for a softmax-like classifier output but not
// guaranteed in general -- bars simply clip outside that range rather than
// misrender). Float32 tensors' "values" line is used the same way.
export function renderLatest(state) {
  if (state.status === null) {
    return '<p class="note">no inference reported yet -- connect, then wait for the board\'s next cycle.</p>';
  }
  const statusColor = state.status === "OK" ? "#2a9d3f" : "#c0392b";
  const rows = [...state.outputs.entries()]
    .map(([index, out]) => {
      const raw = out.raw ? out.raw.join(", ") : "";
      const barValues = out.dequant || out.values || [];
      const bars = barValues
        .map((v) => {
          const pct = Math.max(0, Math.min(100, v * 100));
          return `<div style="display: flex; align-items: center; gap: 0.4em; margin: 1px 0;">
            <span style="width: 5em; text-align: right; font-family: monospace;">${esc(v.toFixed(4))}</span>
            <div style="background: #eee; flex: 1; height: 0.9em;">
              <div style="background: #4a90d9; height: 100%; width: ${pct}%;"></div>
            </div>
          </div>`;
        })
        .join("");
      return `<div style="margin: 0.5em 0;">
        <div class="note">out[${index}]${out.raw ? " raw: " + esc(raw) : ""}</div>
        ${bars}
      </div>`;
    })
    .join("");
  return `
    <p style="margin: 0.3em 0;">
      <b>test inference:</b>
      <span style="color: ${statusColor}; font-weight: 600;">${esc(state.status)}</span>
      (${state.latencyMs} ms)
    </p>
    ${rows}`;
}

function initCardputerPanel() {
  const connectButton = document.getElementById("cardputer-connect-button");
  if (!connectButton) return; // not on this page

  if (!("serial" in navigator)) {
    const unsupportedEl = document.getElementById("cardputer-serial-unsupported");
    if (unsupportedEl) unsupportedEl.style.display = "";
    connectButton.disabled = true;
    return;
  }

  const disconnectButton = document.getElementById("cardputer-disconnect-button");
  const clearButton = document.getElementById("cardputer-clear-button");
  const statusEl = document.getElementById("cardputer-status");
  const latestEl = document.getElementById("cardputer-latest");
  const logEl = document.getElementById("cardputer-log");

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };
  const appendLog = (line) => {
    if (!logEl) return;
    logEl.value += line + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  };

  const state = { status: null, latencyMs: 0, outputs: new Map() };
  let port = null;
  let reader = null;

  async function readLoop() {
    let lineBuffer = "";
    const decoder = new TextDecoder();
    let disconnectReason = null;
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        lineBuffer += decoder.decode(value, { stream: true });
        let newlineIndex;
        while ((newlineIndex = lineBuffer.indexOf("\n")) >= 0) {
          const line = lineBuffer.slice(0, newlineIndex).replace(/\r$/, "");
          lineBuffer = lineBuffer.slice(newlineIndex + 1);
          if (line.length === 0) continue;
          appendLog(line);
          if (ingestLine(state, line) && latestEl) {
            latestEl.innerHTML = renderLatest(state);
          }
        }
      }
    } catch (err) {
      // This board's native USB-Serial/JTAG peripheral disconnects and
      // re-enumerates on every hardware reset -- an expected condition
      // here, not a fatal error (see the panel's own "details").
      disconnectReason = err && err.message ? err.message : String(err);
    } finally {
      try {
        reader.releaseLock();
      } catch {
        /* already released by cancel() */
      }
      reader = null;
      try {
        await port.close();
      } catch {
        /* already closed/disconnected */
      }
      port = null;
      setStatus(
        disconnectReason
          ? `disconnected (${disconnectReason}) -- click Connect to resume`
          : "disconnected -- click Connect to resume",
      );
      connectButton.disabled = false;
      disconnectButton.disabled = true;
    }
  }

  connectButton.addEventListener("click", async () => {
    connectButton.disabled = true;
    try {
      port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      reader = port.readable.getReader();
      setStatus("connected -- streaming");
      disconnectButton.disabled = false;
      readLoop(); // not awaited: runs until the port closes or errors
    } catch (err) {
      setStatus("connect failed: " + (err && err.message ? err.message : err));
      connectButton.disabled = false;
    }
  });

  disconnectButton.addEventListener("click", async () => {
    disconnectButton.disabled = true;
    if (reader) {
      try {
        await reader.cancel();
      } catch (err) {
        setStatus("disconnect error: " + (err && err.message ? err.message : err));
      }
    }
    // readLoop()'s own finally block does the rest (release, close, status).
  });

  if (clearButton) {
    clearButton.addEventListener("click", () => {
      if (logEl) logEl.value = "";
      state.status = null;
      state.outputs = new Map();
      if (latestEl) latestEl.innerHTML = "";
    });
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  initCardputerPanel();
}
