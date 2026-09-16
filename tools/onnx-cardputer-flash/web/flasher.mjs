// Web Serial flashing panel, using Espressif's own esptool-js
// (https://github.com/espressif/esptool-js) — the ROM serial bootloader
// protocol ESP32-S3 (and every other ESP32 variant) speaks, ported to
// JS/Web Serial. This is the piece that makes "flash straight from the
// browser" real rather than aspirational: no driver, no native app, no
// server in the loop, matching the design goal in ../README.md.
//
// NOT covered here: getting from an ONNX model to a flashable image
// (../scripts/onnx_to_tflite_micro.py + ../firmware/README.md) — this panel
// only takes a pre-built merged .bin and writes it to the board.
//
// Status: implemented against esptool-js's documented API (its README's
// usage example), not yet exercised against a real board from this
// environment — there is no attached hardware here to flash. Try it against
// a real Cardputer/ESP32-S3 before relying on it.

const ESPTOOL_JS_VERSION = "0.6.1";
const ESPTOOL_JS_ESM = `https://cdn.jsdelivr.net/npm/esptool-js@${ESPTOOL_JS_VERSION}/lib/index.js`;

const logEl = document.getElementById("log");
const log = (msg) => {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
};

const connectButton = document.getElementById("connect-button");
const disconnectButton = document.getElementById("disconnect-button");
const chipStatus = document.getElementById("chip-status");
const firmwareInput = document.getElementById("firmware-input");
const addressInput = document.getElementById("flash-address");
const flashButton = document.getElementById("flash-button");

if (!("serial" in navigator)) {
  document.getElementById("serial-unsupported").style.display = "";
  document.getElementById("serial-controls").style.display = "none";
} else {
  main();
}

function main() {
  let esploader = null;
  let transport = null;

  // esptool-js's own terminal sink: just routes into this page's log box.
  const terminal = {
    clean: () => { logEl.textContent = ""; },
    writeLine: (data) => log(data),
    write: (data) => log(data),
  };

  connectButton.addEventListener("click", async () => {
    connectButton.disabled = true;
    try {
      const { ESPLoader, Transport } = await import(/* @vite-ignore */ ESPTOOL_JS_ESM);
      const port = await navigator.serial.requestPort();
      transport = new Transport(port, true);
      esploader = new ESPLoader({ transport, baudrate: 115200, terminal });
      log("connecting...");
      const chipName = await esploader.main();
      chipStatus.textContent = `connected: ${chipName}`;
      log(`connected to ${chipName}`);
      disconnectButton.disabled = false;
      firmwareInput.disabled = false;
      flashButton.disabled = false;
    } catch (e) {
      log("connect failed: " + (e && e.message ? e.message : e));
      connectButton.disabled = false;
    }
  });

  disconnectButton.addEventListener("click", async () => {
    try {
      if (transport) await transport.disconnect();
    } catch (e) {
      log("disconnect error: " + (e && e.message ? e.message : e));
    }
    esploader = null;
    transport = null;
    chipStatus.textContent = "";
    connectButton.disabled = false;
    disconnectButton.disabled = true;
    firmwareInput.disabled = true;
    flashButton.disabled = true;
  });

  flashButton.addEventListener("click", async () => {
    if (!esploader) {
      log("not connected");
      return;
    }
    const file = firmwareInput.files && firmwareInput.files[0];
    if (!file) {
      log("pick a firmware .bin first");
      return;
    }
    let address;
    try {
      address = parseInt(addressInput.value, 16);
      if (!Number.isFinite(address) || address < 0) throw new Error("bad address");
    } catch {
      log(`invalid flash address: ${addressInput.value}`);
      return;
    }

    flashButton.disabled = true;
    connectButton.disabled = true;
    disconnectButton.disabled = true;
    try {
      const data = new Uint8Array(await file.arrayBuffer());
      log(`flashing ${file.name} (${data.length.toLocaleString()} bytes) at 0x${address.toString(16)}...`);
      await esploader.writeFlash({
        fileArray: [{ data, address }],
        flashMode: "keep",
        flashFreq: "keep",
        flashSize: "keep",
        compress: true,
        reportProgress: (fileIndex, written, total) => {
          const pct = total > 0 ? Math.floor((written / total) * 100) : 0;
          chipStatus.textContent = `flashing... ${pct}%`;
        },
      });
      log("flash complete, resetting board...");
      await esploader.after("hard_reset");
      log("done — the board should now be running the new firmware.");
      chipStatus.textContent = "flashed — board reset";
    } catch (e) {
      log("flash failed: " + (e && e.message ? e.message : e));
    } finally {
      flashButton.disabled = false;
      disconnectButton.disabled = false;
    }
  });
}
