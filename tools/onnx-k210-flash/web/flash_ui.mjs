import { K210Loader } from "./k210_isp.mjs";

const logEl = document.getElementById("log");
const log = (msg) => {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
};

if (!("serial" in navigator)) {
  document.getElementById("serial-unsupported").style.display = "";
  document.getElementById("controls").style.display = "none";
}

const flashButton = document.getElementById("flash-button");
const firmwareInput = document.getElementById("firmware-input");
const chipTypeSelect = document.getElementById("chip-type");
const resetSchemeSelect = document.getElementById("reset-scheme");
const flashAddressInput = document.getElementById("flash-address");
const skipEraseCheckbox = document.getElementById("skip-erase");

flashButton.addEventListener("click", async () => {
  const file = firmwareInput.files && firmwareInput.files[0];
  if (!file) {
    log("pick a firmware .bin first");
    return;
  }

  flashButton.disabled = true;
  let loader;
  try {
    log("requesting serial port...");
    const port = await navigator.serial.requestPort();
    const resetScheme = resetSchemeSelect ? resetSchemeSelect.value : "dan";
    log(`using reset scheme: ${resetScheme}`);
    loader = new K210Loader(port, { log, resetScheme });
    await loader.connect();

    log("fetching ISP flash-mode stub (isp_stub.bin, vendored from kflash.py)...");
    const stubResp = await fetch("./isp_stub.bin");
    if (!stubResp.ok) throw new Error(`could not fetch isp_stub.bin: HTTP ${stubResp.status}`);
    const stubBytes = new Uint8Array(await stubResp.arrayBuffer());

    const firmwareBytes = new Uint8Array(await file.arrayBuffer());
    const chipType = parseInt(chipTypeSelect.value, 10);
    const addressOffset = flashAddressInput ? parseInt(flashAddressInput.value, 16) : 0;
    if (!Number.isFinite(addressOffset) || addressOffset < 0) {
      throw new Error(`invalid flash address: ${flashAddressInput.value}`);
    }
    const skipErase = skipEraseCheckbox ? skipEraseCheckbox.checked : false;

    log(`flashing ${file.name} (${firmwareBytes.length.toLocaleString()} bytes) at 0x${addressOffset.toString(16)}` +
        (skipErase ? " (skipping erase)" : " (full-chip erase first)") + "...");
    await loader.flashFirmware(stubBytes, firmwareBytes, {
      chipType,
      addressOffset,
      skipErase,
      onStage: (stage) => log(`-- ${stage}`),
      onProgress: (kind, n, total) => {
        if (n === total || n % 8 === 0) log(`  ${kind}: ${n}/${total} chunks`);
      },
    });
    log("done -- the board should now be booting the new firmware.");
  } catch (e) {
    log("flash failed: " + (e && e.message ? e.message : e));
  } finally {
    if (loader) {
      try {
        await loader.disconnect();
      } catch {
        // ignore
      }
    }
    flashButton.disabled = false;
  }
});
