# Cardputer firmware

**Recommended: use the prebuilt generic runtime** — see
[`runtime/README.md`](runtime/README.md). It's a real, compiled,
checksummed image (`runtime/prebuilt/cardputer-runtime.bin`): flash it once
at `0x0`, and every model swap after that is just a second Web Serial write
of a `.tflite` file's bytes at a fixed offset (`0x310000`) — no PlatformIO,
no compiler, no rebuild, ever, in that loop. It is still **not verified
against real hardware** (no Cardputer attached to the environment this was
built in) — the ELF links and sizes fit, but nobody has powered on a board
with it yet.

The rest of this page is the older alternative: baking one specific model
directly into the firmware as a C array, producing a single self-contained
binary with no separate "model" partition. Use it if you specifically want
that (e.g. no interest in ever swapping models on this device), otherwise
prefer the runtime above.

## Why audio (keyword spotting), not vision

Cardputer has no camera — see the top of this conversation's design
discussion. It does have a built-in microphone (SPM1423), so the natural
first model class is audio keyword-spotting / wake-word detection: TFLite
Micro's own reference example (`micro_speech`) is exactly this shape, and
that's the model class `../scripts/onnx_to_tflite_micro.py` is written for.

## 1. PlatformIO project

```ini
; platformio.ini
[env:cardputer]
platform = espressif32
board = esp32-s3-devkitc-1   ; generic ESP32-S3 devkit; Cardputer is ESP32-S3FN8
framework = arduino
board_build.flash_size = 8MB
board_build.partitions = huge_app.csv   ; app needs room for TFLite Micro + model
lib_deps =
    m5stack/M5Cardputer                   ; Cardputer's keyboard/screen/mic HAL (confirmed on the PlatformIO registry)
    spaziochirale/Chirale_TensorFLowLite   ; TFLite Micro, generated from the upstream tflite-micro sources (confirmed buildable -- see runtime/)
```

There is no dedicated PlatformIO board id for Cardputer as of this writing —
`esp32-s3-devkitc-1` plus the flash/partition overrides above is the usual
substitute the M5Stack community uses; verify pin mappings against
`M5Unified`'s own Cardputer board definition before wiring the mic.

## 2. Bake the model in

Run `../scripts/onnx_to_tflite_micro.py your_model.onnx model_data.h` (after
simplifying/quantizing it with the onnxsim converter — see the top-level
`../README.md`), then drop the generated `model_data.h` next to your sketch
and `#include` it in place of the reference example's own model array:

```cpp
#include "model_data.h"   // g_model[], g_model_len

const tflite::Model* model = tflite::GetModel(g_model);
```

Everything else — the `tflite::MicroInterpreter` setup, op resolver, and the
mic-capture loop — follows TFLite Micro's own `micro_speech` example
(https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech)
almost unchanged (`runtime/src/main.cpp` has a real, compiled version of the
`MicroInterpreter`/`AllOpsResolver` setup to copy from); swap its
`AudioProvider` for `M5Cardputer.Mic` reads.

## 3. Build a single flashable image

`onnx-cardputer-flash`'s Web Serial panel (`../web/`) flashes one merged
image at address `0x0` — PlatformIO doesn't merge bootloader + partition
table + app by default, so do it explicitly after `pio run`:

```sh
pio run -e cardputer
esptool.py --chip esp32s3 merge_bin -o merged-firmware.bin \
    --flash_mode qio --flash_freq 80m --flash_size 8MB \
    0x0     .pio/build/cardputer/bootloader.bin \
    0x8000  .pio/build/cardputer/partitions.bin \
    0x10000 .pio/build/cardputer/firmware.bin
```

`merged-firmware.bin` is what you pick in the Web Serial panel's file input,
flashed at `0x0`.

`--flash_mode qio --flash_freq 80m` (not `dio`/`40m`, this section's own
earlier guess): confirmed against `esp32-s3-devkitc-1`'s actual PlatformIO
board manifest (`f_flash: 80000000L, flash_mode: qio`) while building
`runtime/`, not assumed — a mismatched flash mode in the merged image's
header is exactly the kind of thing that compiles fine and then fails
silently on real hardware.
