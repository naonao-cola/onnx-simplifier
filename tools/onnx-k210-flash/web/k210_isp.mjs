// A Web Serial port of the Kendryte K210's ISP flashing protocol -- the
// piece onnx-cardputer-flash's README flagged as missing: unlike ESP32
// (esptool-js, a maintained JS library from Espressif), K210 has no
// existing browser-side flasher, only the Python reference tool
// (https://github.com/kendryte/kflash.py, MIT, Copyright (c) 2019 Kendryte).
//
// This is a protocol port, not a guess: every constant, packet layout, and
// sequence below was read directly out of kflash.py's source (fetched and
// inspected line-by-line while writing this -- see the comments citing
// specific behavior) rather than reconstructed from memory. The one binary
// asset this needs -- the "flash mode" stub K210's mask-ROM ISP loads into
// SRAM and boots into, since the ROM ISP itself can only read/write SRAM
// and cannot touch SPI flash at all -- is vendored byte-for-byte from
// kflash.py's own embedded copy (`isp_stub.bin`, extracted by decompressing
// kflash.py's `ISP_PROG` constant; see ../THIRD_PARTY_NOTICES.md).
//
// Status: reviewed against kflash.py's source, exercised only by the
// framing-level unit tests in ../tests/ (which check byte-exact output
// against kflash.py's own literal packets) -- NOT run against a real
// K210 board. There is no such board attached to the environment this was
// written in. Try it against a real M5StickV / Maix Amigo before relying
// on it, and please report back anything that doesn't match.
//
// Deliberately NOT implemented (see ../README.md for the reasoning):
// the flash-mode baud-rate bump (kflash.py's change_baudrate) -- Web
// Serial has no in-place baud change, only close+reopen, and getting that
// transition wrong loses the connection with no way to verify from here.
// Everything below runs at the ISP's fixed 115200.

const ISP_BAUD = 115200;
const ISP_RECEIVE_TIMEOUT_MS = 1000; // kflash.py's ISP_RECEIVE_TIMEOUT, generously rounded up for browser scheduling
const MAX_RETRY_TIMES = 10; // kflash.py's MAX_RETRY_TIMES
const GREETING_MAX_RETRY = 15; // kflash.py's retry_count > 15 in the greeting loop
const SRAM_STUB_ADDRESS = 0x80000000; // kflash.py's install_flash_bootloader() / boot() default
const MEM_WRITE_CHUNK = 1024; // kflash.py's flash_dataframe() DATAFRAME_SIZE
const FLASH_WRITE_CHUNK = 4096 * 16; // kflash.py's ISP_FLASH_DATA_FRAME_SIZE (4096-byte sector * 16)

// ISPResponse.ISPOperation / FlashModeResponse.Operation in kflash.py.
const ISP_OP = {
  ECHO: 0xc1,
  NOP: 0xc2,
  MEMORY_WRITE: 0xc3,
  MEMORY_READ: 0xc4,
  MEMORY_BOOT: 0xc5,
  DEBUG_INFO: 0xd1, // shared with flash mode
  CHANGE_BAUDRATE: 0xc6,
};
const FLASH_OP = {
  DEBUG_INFO: 0xd1,
  NOP: 0xd2,
  FLASH_ERASE: 0xd3,
  FLASH_WRITE: 0xd4,
  REBOOT: 0xd5,
  BAUDRATE_SET: 0xd6,
  FLASH_INIT: 0xd7,
};
const RET_OK = 0xe0;

// greeting()/flash_greeting()/flash_erase() in kflash.py send a hardcoded
// raw literal frame directly (`self._port.write(...)`), bypassing its own
// packet-builder entirely -- a different, shorter header shape (op byte +
// 12 zero bytes, 15 bytes total incl. SLIP delimiters) than every other
// command (which is op(u16) + reserved(u16) + crc32(u32) + body via
// `self.write()`). Reproducing these three via buildPacket() would send
// the wrong bytes, so they're copied verbatim instead.
export const GREETING_FRAME = new Uint8Array([0xc0, 0xc2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xc0]);
export const FLASH_GREETING_FRAME = new Uint8Array([0xc0, 0xd2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xc0]);
export const FLASH_ERASE_FRAME = new Uint8Array([0xc0, 0xd3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xc0]);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// DTR/RTS reset sequences, [dtr, rts] per step, 100ms apart -- kflash.py's
// reset_to_isp_*()/reset_to_boot_*() for the three board families most
// likely to cover M5StickV/Maix Amigo (plain CH340/CP210x-style boards;
// "goE"/"trainer"/"bit_mic" need FTDI dual-interface port auto-detection
// that doesn't apply here, so they're not included). "dan" is the default
// -- Sipeed's own "Dan Dock" scheme, what community M5StickV instructions
// use -- but which one a given board actually needs is exactly the kind
// of thing that only shows up against real hardware; if "dan" doesn't
// enter ISP mode, try the other two before assuming something else is
// wrong.
export const RESET_SCHEMES = {
  dan: {
    isp: [[false, false], [false, true], [true, false]],
    boot: [[false, false], [false, true], [false, false]],
  },
  kd233: {
    isp: [[false, false], [true, false], [false, true]],
    boot: [[false, false], [true, false], [false, false]],
  },
  goD: {
    isp: [[true, true], [true, false], [true, false]],
    boot: [[false, false], [true, false], [true, true]],
  },
};

// Standard CRC-32 (zlib/PNG/gzip polynomial, reflected 0xEDB88320) --
// matches Python's binascii.crc32, which is what every packet's checksum
// field in kflash.py is computed with.
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC32_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function concatBytes(...parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

function u16le(n) {
  return new Uint8Array([n & 0xff, (n >> 8) & 0xff]);
}
function u32le(n) {
  return new Uint8Array([n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >>> 24) & 0xff]);
}

// Builds one ISP request packet's body (everything a SLIP frame carries),
// matching kflash.py's own construction exactly: op(u16) + reserved(u16, 0)
// + crc32-of-body(u32) + body. The CRC covers `body` alone, never the
// op/reserved/crc header itself (verified against kflash.py's boot()/
// flash_dataframe()/etc., which all crc32() only the pre-header bytes).
export function buildPacket(op, body = new Uint8Array(0)) {
  const crc = u32le(crc32(body));
  return concatBytes(u16le(op), u16le(0), crc, body);
}

// SLIP-encodes a packet for the wire: 0xc0-delimited, with 0xc0 -> 0xdb 0xdc
// and 0xdb -> 0xdb 0xdd inside the frame (kflash.py's MAIXLoader.write()).
export function slipEncode(packet) {
  const out = [0xc0];
  for (const b of packet) {
    if (b === 0xc0) out.push(0xdb, 0xdc);
    else if (b === 0xdb) out.push(0xdb, 0xdd);
    else out.push(b);
  }
  out.push(0xc0);
  return new Uint8Array(out);
}

// Incremental SLIP frame decoder -- mirrors kflash.py's slip_reader()
// generator, but fed chunks from Web Serial's read loop instead of
// blocking reads. feed() returns an array of any frames it completed.
class SlipDecoder {
  constructor() {
    this._partial = null; // null = waiting for a frame's opening 0xc0
    this._inEscape = false;
  }
  feed(chunk) {
    const frames = [];
    for (const b of chunk) {
      if (this._partial === null) {
        if (b === 0xc0) this._partial = [];
        // else: garbage before a frame start -- kflash.py raises; we just skip it.
        continue;
      }
      if (this._inEscape) {
        this._inEscape = false;
        if (b === 0xdc) this._partial.push(0xc0);
        else if (b === 0xdd) this._partial.push(0xdb);
        // else: invalid escape -- kflash.py raises; we drop the frame instead.
        else this._partial = null;
        continue;
      }
      if (b === 0xdb) {
        this._inEscape = true;
      } else if (b === 0xc0) {
        frames.push(new Uint8Array(this._partial));
        this._partial = null;
      } else {
        this._partial.push(b);
      }
    }
    return frames;
  }
}

// Parses a decoded ISP/flash-mode response frame: op(u8), reason(u8),
// optional ASCII debug text (kflash.py's ISPResponse.parse / FlashModeResponse.parse
// -- note the *response* header is 2 bytes, not the 4-byte op+reserved the
// *request* side uses).
function parseResponse(frame) {
  if (frame.length < 2) throw new Error(`short K210 response frame (${frame.length} bytes)`);
  const op = frame[0];
  const reason = frame[1];
  const text = frame.length > 2 ? new TextDecoder().decode(frame.subarray(2)) : "";
  return { op, reason, text };
}

function chunk(bytes, size) {
  const out = [];
  for (let i = 0; i < bytes.length; i += size) out.push(bytes.subarray(i, i + size));
  return out;
}

export class K210Loader {
  constructor(port, { log = () => {}, resetScheme = "dan" } = {}) {
    if (!RESET_SCHEMES[resetScheme]) {
      throw new Error(`unknown resetScheme ${resetScheme} -- one of ${Object.keys(RESET_SCHEMES).join(", ")}`);
    }
    this.port = port;
    this.log = log;
    this.resetScheme = resetScheme;
    this._decoder = new SlipDecoder();
    this._frameQueue = [];
    this._waiters = [];
    this._writer = null;
    this._readLoopPromise = null;
    this._closed = false;
  }

  // --- transport -----------------------------------------------------------

  async connect() {
    await this.port.open({ baudRate: ISP_BAUD, dataBits: 8, stopBits: 1, parity: "none" });
    this._writer = this.port.writable.getWriter();
    this._readLoopPromise = this._readLoop();
  }

  async disconnect() {
    this._closed = true;
    try {
      if (this._writer) {
        await this._writer.close();
        this._writer = null;
      }
    } catch {
      // ignore -- port may already be gone
    }
    try {
      if (this.port.readable) await this.port.readable.cancel();
    } catch {
      // ignore
    }
    try {
      await this._readLoopPromise;
    } catch {
      // ignore
    }
    try {
      await this.port.close();
    } catch {
      // ignore
    }
  }

  async _readLoop() {
    const reader = this.port.readable.getReader();
    try {
      while (!this._closed) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value && value.length) {
          for (const frame of this._decoder.feed(value)) this._deliverFrame(frame);
        }
      }
    } catch {
      // port closed/disconnected -- fall through
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // ignore
      }
    }
  }

  _deliverFrame(frame) {
    if (this._waiters.length) this._waiters.shift()(frame);
    else this._frameQueue.push(frame);
  }

  _readFrame(timeoutMs = ISP_RECEIVE_TIMEOUT_MS) {
    if (this._frameQueue.length) return Promise.resolve(this._frameQueue.shift());
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const idx = this._waiters.indexOf(waiter);
        if (idx >= 0) this._waiters.splice(idx, 1);
        reject(new Error("timed out waiting for a response from the board"));
      }, timeoutMs);
      const waiter = (frame) => {
        clearTimeout(timer);
        resolve(frame);
      };
      this._waiters.push(waiter);
    });
  }

  async _send(op, body) {
    await this._writer.write(slipEncode(buildPacket(op, body)));
  }

  async _setSignals(dtr, rts) {
    await this.port.setSignals({ dataTerminalReady: dtr, requestToSend: rts });
  }

  // --- reset sequences -------------------------------------------------
  // Runs this.resetScheme's steps (see RESET_SCHEMES) -- 100ms per step,
  // matching kflash.py's own reset_to_isp_*()/reset_to_boot_*() timing.

  async resetToISP() {
    for (const [dtr, rts] of RESET_SCHEMES[this.resetScheme].isp) {
      await this._setSignals(dtr, rts);
      await sleep(100);
    }
  }

  async resetToBoot() {
    for (const [dtr, rts] of RESET_SCHEMES[this.resetScheme].boot) {
      await this._setSignals(dtr, rts);
      await sleep(100);
    }
  }

  // --- mask-ROM ISP stage ------------------------------------------------

  // One greeting attempt: send, wait once, done -- no retry of its own
  // (kflash.py's greeting() doesn't retry internally either; it's the
  // *caller* that retries, pairing each attempt with a fresh reset pulse
  // below, since a single reset may just not have taken).
  async greeting() {
    await this._writer.write(GREETING_FRAME);
    const { op } = parseResponse(await this._readFrame(500));
    if (op !== ISP_OP.NOP) throw new Error(`unexpected greeting response op 0x${op.toString(16)}`);
  }

  // Re-asserts the ISP reset sequence before each greeting attempt (kflash.py's
  // main retry loop: reset_to_isp_dan() + greeting(), up to 15 times) --
  // greeting() alone won't recover from a reset that didn't take.
  async enterISPMode() {
    for (let attempt = 0; attempt < GREETING_MAX_RETRY; attempt++) {
      await this.resetToISP();
      try {
        await this.greeting();
        return;
      } catch {
        // this attempt's reset pulse didn't take (or the response timed out) -- retry
      }
    }
    throw new Error("no K210 found: greeting failed after retrying reset");
  }

  // Uploads `stubBytes` to SRAM at 0x80000000 via repeated ISP_MEMORY_WRITE
  // calls (kflash.py's flash_dataframe(), 1KiB chunks, retried on any
  // non-OK/non-default reason), then jumps to it with ISP_MEMORY_BOOT.
  async installStub(stubBytes, { onProgress = () => {} } = {}) {
    const chunks = chunk(stubBytes, MEM_WRITE_CHUNK);
    let address = SRAM_STUB_ADDRESS;
    for (let i = 0; i < chunks.length; i++) {
      const body = concatBytes(u32le(address), u32le(chunks[i].length), chunks[i]);
      let ok = false;
      for (let attempt = 0; attempt <= MAX_RETRY_TIMES && !ok; attempt++) {
        await this._send(ISP_OP.MEMORY_WRITE, body);
        try {
          const { reason } = parseResponse(await this._readFrame());
          ok = reason === RET_OK || reason === 0; // ISP_RET_DEFAULT or ISP_RET_OK
        } catch {
          // timeout -- retry this chunk
        }
      }
      if (!ok) throw new Error(`stub upload failed at chunk ${i}/${chunks.length}`);
      address += chunks[i].length;
      onProgress(i + 1, chunks.length);
    }
    await this._send(ISP_OP.MEMORY_BOOT, concatBytes(u32le(SRAM_STUB_ADDRESS), u32le(0)));
  }

  // --- flash-mode stage (implemented by the uploaded stub) ---------------

  // Polls for the stub's own greeting -- it takes a moment to boot after
  // ISP_MEMORY_BOOT, so this genuinely needs the retry loop kflash.py uses
  // (flash_greeting()), not just one shot.
  async flashGreeting() {
    for (let attempt = 0; attempt <= MAX_RETRY_TIMES; attempt++) {
      try {
        await this._writer.write(FLASH_GREETING_FRAME);
        const { op, reason } = parseResponse(await this._readFrame());
        if (op === FLASH_OP.NOP && reason === RET_OK) return;
      } catch {
        // timeout -- retry
      }
      await sleep(100);
    }
    throw new Error("failed to connect to K210's flash-mode stub");
  }

  // chipType: 0 = in-chip flash (the K210 module's own SPI flash -- what
  // M5StickV/Maix Amigo have), 1 = on-board flash (a separate flash chip
  // wired to the K210, on some dev boards).
  async initFlash(chipType = 0) {
    const body = concatBytes(u32le(chipType), u32le(0));
    for (let attempt = 0; attempt <= MAX_RETRY_TIMES; attempt++) {
      await this._send(FLASH_OP.FLASH_INIT, body);
      try {
        const { op, reason } = parseResponse(await this._readFrame());
        if (op === FLASH_OP.FLASH_INIT && reason === RET_OK) return;
      } catch {
        // timeout -- retry
      }
    }
    throw new Error("failed to initialize K210 flash");
  }

  async flashErase() {
    await this._writer.write(FLASH_ERASE_FRAME);
    const { reason } = parseResponse(await this._readFrame(10000)); // full-chip erase is slow
    if (reason !== RET_OK) throw new Error(`flash erase failed (reason 0x${reason.toString(16)})`);
  }

  // Writes `firmwareBytes` to flash starting at `addressOffset`, framed
  // exactly as kflash.py's flash_firmware() does: [aes_flag=0x00][len u32 LE]
  // [firmware bytes][SHA-256 of the three previous fields] -- kflash.py
  // computes that hash with Python's hashlib.sha256; this uses the
  // browser's Web Crypto SubtleCrypto.digest for the same SHA-256, not a
  // vendored implementation. The result is split into 64KiB chunks
  // (zero-padded on the last one) and each chunk is written with
  // FLASH_WRITE (0xd4) at chunk_index * 64KiB + addressOffset.
  async writeFirmware(firmwareBytes, { addressOffset = 0, onProgress = () => {} } = {}) {
    const aesFlag = new Uint8Array([0x00]); // no AES encryption
    const lenField = u32le(firmwareBytes.length);
    const withoutHash = concatBytes(aesFlag, lenField, firmwareBytes);
    const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", withoutHash));
    const withHeader = concatBytes(withoutHash, hash);

    const chunks = chunk(withHeader, FLASH_WRITE_CHUNK);
    for (let i = 0; i < chunks.length; i++) {
      let padded = chunks[i];
      if (padded.length < FLASH_WRITE_CHUNK) {
        const p = new Uint8Array(FLASH_WRITE_CHUNK);
        p.set(padded);
        padded = p;
      }
      const address = i * FLASH_WRITE_CHUNK + addressOffset;
      const body = concatBytes(u32le(address), u32le(padded.length), padded);
      let ok = false;
      for (let attempt = 0; attempt <= MAX_RETRY_TIMES && !ok; attempt++) {
        await this._send(FLASH_OP.FLASH_WRITE, body);
        try {
          const { reason } = parseResponse(await this._readFrame(3000)); // kflash.py widens this timeout for the actual write loop
          ok = reason === RET_OK;
        } catch {
          // timeout -- retry this chunk
        }
      }
      if (!ok) throw new Error(`flash write failed at chunk ${i}/${chunks.length}`);
      onProgress(i + 1, chunks.length);
    }
  }

  // Runs the whole sequence in the order kflash.py's own process() does:
  // enter ISP mode, upload+boot the flash-mode stub, wait for it to greet,
  // init(+erase) flash, write the firmware, then reset into it. `stubBytes`
  // is normally isp_stub.bin fetched alongside this module.
  //
  // `skipErase` (default false -- erase runs): FLASH_ERASE (0xd3) has no
  // address/range parameter at the protocol level -- kflash.py's own
  // flash_erase() sends the same fixed, args-less frame this SDK's
  // FLASH_ERASE_FRAME does -- so it is a **full-chip** erase, not a
  // range erase. Calling flashFirmware() twice at two different
  // addressOffsets (e.g. a runtime image at 0x0, then a model at
  // 0x00C00000, per onnx-k210-flash/firmware/runtime/README.md's flash
  // layout) with the default skipErase=false would erase the first
  // write before the second one's write ever happens. skipErase=true
  // writes without erasing first -- correct only if the flash-mode
  // stub's own FLASH_WRITE (0xd4) implementation erases the sectors it's
  // about to program itself before writing them (a common convenience in
  // embedded NOR-flash write helpers, but NOT verified here: this
  // session has no K210 board to check it against, and the ISP stub is
  // an opaque vendored binary with no available source -- see
  // ../README.md's "Flashing a model without re-erasing" section before
  // relying on this for anything other than a first careful hardware
  // test).
  async flashFirmware(stubBytes, firmwareBytes, { chipType = 0, addressOffset = 0, skipErase = false, onStage = () => {}, onProgress = () => {} } = {}) {
    onStage("entering ISP mode");
    await this.enterISPMode();
    onStage("uploading flash-mode stub");
    await this.installStub(stubBytes, { onProgress: (n, total) => onProgress("stub", n, total) });
    onStage("waiting for flash-mode stub");
    await this.flashGreeting();
    onStage("initializing flash");
    await this.initFlash(chipType);
    if (!skipErase) {
      onStage("erasing flash (full chip -- see flashFirmware's own comment)");
      await this.flashErase();
    }
    onStage("writing firmware");
    await this.writeFirmware(firmwareBytes, { addressOffset, onProgress: (n, total) => onProgress("firmware", n, total) });
    onStage("resetting into new firmware");
    await this.resetToBoot();
  }
}
