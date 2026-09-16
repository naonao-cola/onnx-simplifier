# onnx-cardputer-flash

Browser demo: take an ONNX model straight from Hugging Face, simplify/quantize
it with onnxsim, convert it for TFLite Micro, and flash the resulting
firmware onto an ESP32-S3 board (M5Stack Cardputer) — no native app, no
server, no driver install, using
[Web Serial](https://developer.chrome.com/docs/capabilities/serial) instead
of a USB-DFU/WebUSB path (see this tool's design discussion for why Web
Serial was picked over WebUSB for this class of board).

## Why Cardputer, and why Web Serial

- **Web Serial, not WebUSB**: ESP32's ROM bootloader speaks a UART-over-USB
  ISP protocol (through the board's USB-serial chip), not raw USB DFU — that
  is exactly what Web Serial is for, and it is a *solved* problem thanks to
  Espressif's own [`esptool-js`](https://github.com/espressif/esptool-js),
  which this tool's flashing panel uses directly. A K210 board (M5StickV,
  Sipeed Maix Amigo) is the same story in principle but has no existing
  Web Serial port of its ISP protocol — a real follow-up, not attempted here.
- **Cardputer**: an ESP32-S3 board, so it gets `esptool-js`'s flashing story
  for free. It has no camera, so the model class this recipe targets is
  audio keyword-spotting (it has a built-in mic) rather than vision — see
  `firmware/README.md`.

## Pipeline and what's verified

| Step | Where | Status |
|---|---|---|
| Fetch + simplify/quantize an HF model | the existing [onnxsim model converter](../convertmodel/index.html) — this tool doesn't repeat that UI | already shipped, unrelated to this addition |
| ONNX → TFLite (int8) | `scripts/onnx_to_tflite_micro.py` (`convert_to_tflite`, wraps `onnx2tf -oiqt`) | **run for real against real HF models — found two real bugs, both worked around manually**: `-oiqt` requires a Float32 ONNX input, so it errors out on an already-quantized model (finding #3); and even given a Float32 input, `-oiqt`'s own default calibration data fails to load on a current numpy for *any* model (finding #6, an onnx2tf bug, not this repo's). Supplying real calibration data directly via onnx2tf's own `-cind` flag (not yet wired into this script) works and produced a real, correctly-quantized int8 model — see `firmware/runtime/README.md`'s "Status", findings #3 and #6 |
| TFLite → C header | `scripts/onnx_to_tflite_micro.py` (`emit_c_header`) | unit-tested, see `tests/` |
| Firmware (generic TFLite Micro runtime) | [`firmware/runtime/`](firmware/runtime/README.md) — a real PlatformIO project, not just a recipe | **compiled, flashed, booted, and ran a real, correctly-quantized model end to end on a real M5Stack Cardputer, now driven by the board's own microphone** -- `Invoke()` returns `kTfLiteOk` on real audio, confirmed live over serial with real output values and latency (findings #6, #7). Getting there found and fixed a `qio`→`dio` flash-mode bug (finding #1); found (unfixed, upstream-library) a crash on legacy-quantized `DepthwiseConv` models (finding #2); found and precisely root-caused (unfixed, upstream-library) that a differently-quantized model's `Transpose` op runs on unsupported `uint8` data (finding #4); and found and fixed a `Serial`-routing bug that was hiding *all* app/TFLM diagnostic output (finding #5) — see that README's "Status" for the full sequence |
| Flash over Web Serial | `web/flasher.mjs` (Espressif's `esptool-js`) | loads and runs its UI logic cleanly in a browser (checked headless); the underlying ISP protocol was exercised for real via `esptool` directly against `/dev/ttyACM0` (same USB-Serial/JTAG port Web Serial would use) — `flasher.mjs`'s own browser-side call shapes are still **not** exercised through an actual Chrome Web Serial session |

Real hardware confirmed, fully end to end: flashing, booting, mmap'ing the
model partition, `AllocateTensors()`, and a correct `Invoke()` all work for
a real, properly int8-quantized model fed real microphone audio through
TF's own audio frontend -- `test inference: OK`, with real (if currently
unremarkable -- no wake word was actually spoken at it) output values and
latency, repeatedly, live, over the now-working `Serial` port (findings #5,
#7). The specific model tried first (legacy-quantized, finding #2/#4)
still doesn't work -- that's a real upstream-library gap, not something
wrong with the pipeline itself. The browser UI itself doing the flashing
(vs. `esptool` CLI standing in for it) is also still unverified. See
`firmware/runtime/README.md`'s "Status" for the full, numbered findings.

## Using it

1. **Once per board:** serve `web/` locally (`python3 -m http.server` from
   `web/`), open it in Chrome/Edge, Connect, pick
   `firmware/runtime/prebuilt/cardputer-runtime.bin`, flash at `0x0`.
2. Simplify/quantize your model with the
   [onnxsim converter](../convertmodel/index.html) (the candidate-model
   table on this tool's own page has one-click links), download the result.
3. `pip install onnx2tf` (pulls in TensorFlow), then:
   ```sh
   python3 scripts/onnx_to_tflite_micro.py your_model.onnx model.tflite
   ```
   (a `.tflite` output path writes the plain converted model; any other
   extension writes a C header instead — the runtime firmware reads a
   plain `.tflite` file from its flash partition, not a C array).
4. Flash *that* `.tflite` file's raw bytes at `0x310000`, same page, same
   Connect session — no rebuild, no PlatformIO, step 1 doesn't repeat.
5. Reset the board (or power-cycle) — it prints what it loaded over serial
   (115200 baud) and on its own screen.

Prefer one self-contained binary per model instead? `firmware/README.md`'s
older recipe bakes the model directly into the firmware as a C array.

## Not done yet / follow-ups

- **Done, real hardware confirmed:** a genuinely per-channel int8-quantized
  model (the HF repo's `-fp32-onnx` variant run through `onnx2tf -oiqt`
  with real calibration data) flashed, booted, and `Invoke()` returned
  `kTfLiteOk` repeatedly -- real, correct end-to-end inference, not just
  "doesn't crash". See `firmware/runtime/README.md`'s finding #6.
- **Wire `onnx2tf`'s `-cind`/custom-calibration-data support into
  `onnx_to_tflite_micro.py`** (findings #3 and #6): `convert_to_tflite()`
  always passes plain `-oiqt`, which (a) errors out on an already-quantized
  ONNX input (finding #3) and (b) even for a Float32 input, only works if
  onnx2tf's own default image-shaped calibration data loads successfully,
  which it currently doesn't for *any* model on a current numpy (finding
  #6) -- both were worked around manually outside the script this session.
  The script should accept caller-supplied calibration data (numpy array or
  path) and pass it through via `-cind`, rather than only ever relying on
  onnx2tf's broken default.
- **Fix `convert_to_tflite()`'s output-file glob** (`tests/` doesn't cover
  this): `sorted(out_dir.glob(f"*{suffix}"))[0]` with
  `suffix = "_integer_quant.tflite"` matches *both*
  `..._integer_quant.tflite` and `..._full_integer_quant.tflite` (the
  latter also ends with that suffix) and picks whichever sorts first
  alphabetically -- currently `full_integer_quant` (the one with real int8
  I/O boundaries, which is what worked on real hardware this session), but
  that's alphabetical luck, not an intentional selection. Should match the
  exact filename onnx2tf documents for `-oiqt`'s int8-with-int8-I/O output,
  not a glob that happens to also catch a differently-named sibling file.
- A Sipeed Maix Amigo / M5StickV (Kendryte K210) target: same board family,
  a real camera (Amigo) and better on-device inference (KPU). The Web
  Serial flasher this needed didn't exist anywhere, so it's now built —
  see [`../onnx-k210-flash/`](../onnx-k210-flash/README.md) (a from-scratch
  port of `kflash.py`'s ISP protocol). Its onnx→kmodel conversion
  (nncase, not TFLite Micro) is done and run for real against a real HF
  model; still needs a firmware recipe (MaixPy or bare-metal K210 SDK) to
  actually load a kmodel on-device, mirroring this tool's `firmware/`.
- No CI coverage — nothing here can run without either a real board or a
  TensorFlow install this repo doesn't otherwise carry.
- Vision models are out of scope for Cardputer specifically (no camera); an
  ESP32-S3 board with a camera module would reuse everything except
  `firmware/README.md`'s mic-specific wiring.
