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
| ONNX → TFLite (int8), via onnx2tf | `scripts/onnx_to_tflite_micro.py` (`convert_to_tflite`, wraps `onnx2tf -oiqt`) | **run for real against real HF models — found two real bugs, both worked around manually**: `-oiqt` requires a Float32 ONNX input, so it errors out on an already-quantized model (finding #3); and even given a Float32 input, `-oiqt`'s own default calibration data fails to load on a current numpy for *any* model (finding #6, an onnx2tf bug, not this repo's). Supplying real calibration data directly via onnx2tf's own `-cind` flag (not yet wired into this script) works and produced a real, correctly-quantized int8 model — see `firmware/runtime/README.md`'s "Status", findings #3 and #6. Needs a real TensorFlow install -- can't run in-browser, see the row below |
| ONNX → TFLite, via flatbuffers directly (no TensorFlow) | `scripts/onnx_to_tflite_flatbuffers.py` | **Run for real against all 5 of this table's own candidate models, output verified against every one.** Recognizes the "fake-quant sandwich" pattern (DequantizeLinear → op → QuantizeLinear) a TF→ONNX exporter produces when re-exporting an *already-quantized* TFLite graph, and repackages the quantized weights/scale/zero-point already in the ONNX file straight into a `.tflite` -- no TensorFlow, no requantization, no new-computed quantization params. Covers CONV_2D/DEPTHWISE_CONV_2D (incl. real per-channel weight quantization and fused ReLU), FULLY_CONNECTED (both `MatMul`+separate-`Add`-bias and 3-input `Gemm`), AVERAGE_POOL_2D, and SOFTMAX -- see that script's own module docstring for exactly which patterns are recognized; it is **not** a general ONNX importer. Every candidate model's emitted `.tflite` loads in a real TFLite interpreter (`ai-edge-litert`) and was checked against `onnx.reference.ReferenceEvaluator` over 100 random trials each: exact match every time for 3 of 5 (TinyConv, Streaming DS-CNN, the Autoencoder); the other 2 (DS-CNN, DS-CNN Large -- both 9-11 conv layers deep) mismatch on ~2% of trials by single-digit counts on an already-saturated output, consistent with accumulated float-vs-fixed-point rounding drift across many layers (same category as TinyConv's own ~1-in-200 single-layer case), not a structural bug -- see `tests/test_onnx_to_tflite_flatbuffers.py`'s module docstring for the full per-model numbers. Not yet wired into a browser UI (see "Not done yet / follow-ups") |
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
   `firmware/runtime/prebuilt/cardputer-runtime.bin`, leave the "what are
   you flashing" dropdown on its default "runtime firmware (0x0)" preset,
   flash.
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

   Model already carries real TFLite-style quantization (like the
   candidate models above) and you'd rather not install TensorFlow?
   `pip install flatbuffers onnx`, then:
   ```sh
   python3 scripts/onnx_to_tflite_flatbuffers.py your_model.onnx model.tflite
   ```
   — no TensorFlow, no onnx2tf. See "Not done yet / follow-ups" below for
   exactly what op patterns this recognizes; it's real but narrower than
   the onnx2tf path.
4. Flash *that* `.tflite` file's raw bytes at `0x310000` — switch the
   dropdown to "model (0x310000)" — same page, same Connect session, no
   rebuild, no PlatformIO, step 1 doesn't repeat.
5. Reset the board (or power-cycle) — it prints what it loaded over serial
   (115200 baud) and on its own screen. Or skip the serial terminal
   entirely: the
   [onnxsim model converter](../convertmodel/index.html#sec-cardputer)'s
   own "Live Cardputer output" panel connects over Web Serial and streams
   the same lines straight into that page, rendering each cycle's status,
   latency, and output values (see
   `../convertmodel/cardputer_monitor_ui.mjs`) — a live monitor, not a
   flasher; it doesn't touch the board's flash.

Building your own firmware and model instead of using the prebuilt runtime
and swappable `.tflite`? Two supported paths, both documented with real
build commands:

- **Custom runtime firmware, same swappable-model layout** (change
  `main.cpp`'s behavior but keep flashing models separately at `0x310000`):
  `firmware/runtime/README.md`'s "Rebuilding it yourself".
- **One self-contained binary with your model baked in** (no separate
  model partition, no swapping later): `firmware/README.md`'s "Bake the
  model in" / "Build a single flashable image".

Either way, the resulting `.bin` still flashes at `0x0` through the same
panel — pick "custom address…" from the dropdown only if you deliberately
want a non-default offset (e.g. testing a custom image at a spare address
before overwriting your known-working runtime).

## Not done yet / follow-ups

- **Emitting `.tflite` with only `flatbuffers` (no TensorFlow): done, and
  now covers all 5 of this table's own candidate models, not just one.**
  `scripts/onnx_to_tflite_flatbuffers.py` recognizes the "fake-quant
  sandwich" a TF→ONNX exporter produces for an already-quantized TFLite
  graph and repackages it directly, without TensorFlow -- verified
  against every real Hugging Face model this tool's own README lists (see
  the pipeline table above and that script's own module docstring). It
  handles CONV_2D/DEPTHWISE_CONV_2D (including real per-channel weight
  quantization -- confirmed directly: every model except TinyConv
  quantizes weights per output-channel, not per-tensor -- and Conv's
  fused ReLU), FULLY_CONNECTED (a bias-less `MatMul`, a `MatMul` +
  separate `Add`-bias node, or a 3-input `Gemm`), AVERAGE_POOL_2D, and
  SOFTMAX. It is **not** a general ONNX importer -- it raises
  `NotImplementedError` rather than emit something silently wrong on
  anything outside the patterns it recognizes. Real follow-ups:
  - **Not wired into a browser UI yet.** Unlike `onnx-k210-flash/web/ncc_wasm.mjs`
    (nncase compiled to WASM, see that tool's README), this script hasn't
    been ported to run in a browser -- it's plain Python today. Since it
    has no TensorFlow/heavy-framework dependency to begin with (that was
    the whole point), this should be a much smaller lift than the nncase
    port was: either a from-scratch JS port of the same pattern-matching +
    `flatbuffers`-npm-package emission logic, or, if Pyodide's own
    `flatbuffers`/`onnx` wheels are good enough, running this exact script
    under Pyodide with no C++/wasm build at all.
  - **PTQ calibration (`--dataset`-style, computing new quantization
    parameters from scratch) is still out of scope by design** -- every
    model this handles already carries real quantization params computed
    upstream; a model that's still float32 going in needs a different,
    genuinely harder tool (this is the same category of gap
    `onnx-k210-flash`'s nncase-wasm path has for real PTQ, not unique to
    this script).
  - **Real op coverage is exactly what these 5 models exercise, not more.**
    Grouped conv (`group` other than 1 or a real per-channel depthwise),
    Gemm with `transA`/`transB` set, multiple graph inputs/outputs, and
    ops these models don't use (Concat, MaxPool, Add/Mul as a residual
    connection rather than a bias, ...) all still raise
    `NotImplementedError` -- extending coverage further needs a real model
    that actually exercises the next op, same as everywhere else in this
    repo.
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
