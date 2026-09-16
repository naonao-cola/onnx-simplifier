# K210 generic kmodel runtime (prebuilt)

The K210 counterpart to `onnx-cardputer-flash/firmware/runtime`:
`prebuilt/onnx-k210-runtime.bin` is a real, compiled, checksummed image.
Flash it once at address `0x0` and it never needs *rebuilding* again --
swapping models afterward means flashing a plain `.kmodel` file's bytes
at a fixed address (`0x00C00000`), same `onnx-k210-flash` flasher, no
toolchain involved. Skipping the erase for that model-only write is now
**verified safe on real hardware** -- see "Flashing a model without
re-erasing" below.

## Status

**Compiled and linked for real**, against `kendryte-standalone-sdk`
(`master`, RISC-V toolchain `kendryte-gnu-toolchain` 8.2.0):

```
onnx_k210_runtime.bin: 1,056,312 bytes
```

**Flashed and booted for real on a Sipeed Maix Amigo.** See
`../../README.md`'s "Testing against real hardware" for the full session
(reset-scheme reliability, flash-addressing gotchas, and how "skip erase"
got verified safe) -- summarized here for the two findings specific to
this runtime's own code:

**Real-hardware finding (confirmed): the blank/erased-model detection path
works exactly as designed.** Flashed fresh, reset, read live over UART:
```
onnx-k210-flash runtime
reading model from flash @0x00c00000 (2097152 bytes)...
no model flashed yet at 0x00c00000 (identifier 0xffffffff, expected 0x4b4d444c)
flash a .kmodel there over Web Serial, then reset.
```
`w25qxx_read_data`'s timing/pin assumptions are real, then: the 2MB read
and the identifier check both complete promptly on real hardware.

**Real-hardware finding (real bug, precisely root-caused, unfixed here):
`interpreter::load_model()` hangs indefinitely given a real, valid
kmodel.** A real Hugging Face model
(`ketiswp/mlcommons-ResNet8-CIFAR10-fp32-onnx`), compiled via
`../../scripts/onnx_to_kmodel.py` to a 104792-byte kmodel (correct `KMDL`
magic, confirmed byte-for-byte) and flashed to `0x00C00000`, made the
runtime print `reading model from flash...` and then go silent -- no
crash, no further output, for 25+ seconds. Root-caused with an
instrumented debug rebuild (extra `printf`s bracketing the identifier
check and the `interp.load_model()` call, from a real
`kendryte-standalone-sdk` checkout, not the version vendored inside
PlatformIO's `framework-maixduino` -- that one is missing `lib/nncase`
entirely): the identifier check passes correctly (`0x4b4d444c`), execution
reaches `interp.load_model()`, and it **never returns** -- the debug
build's own trailing `load_model() returned` print never appeared. Not
root-caused further than that (would need real debugging inside nncase
v1's K210 runtime itself -- KPU peripheral programming, DMA descriptor
setup, etc., none of which this session's instrumentation reached). The
`Simulator`-verified compile (see `../../README.md`'s "Model conversion")
proves the *compiler* output is a well-formed, correctly-executing kmodel
on the host; this proves the *on-device* `interpreter::load_model()` call
specifically is where the real gap is, not the model or the conversion
pipeline.

## How it works

`src/main.cpp`:

1. Reads a fixed `2MB` region of SPI flash starting at `0x00C00000`
   into a RAM buffer via `w25qxx_read_data(..., W25QXX_QUAD_FAST)` --
   **not** memory-mapped. Unlike ESP32-S3 (`esp_partition_mmap`), K210 has
   no flash-mapped execution model at all: the mask ROM copies the whole
   firmware image into K210's 8MB SRAM and runs it from there (see
   `onnx-k210-flash`'s own ISP work -- `SRAM_STUB_ADDRESS 0x80000000` is
   SRAM, not flash), so there's no equivalent "read flash in place"
   mechanism to use instead.
2. Checks the first 4 bytes for nncase's kmodel identifier (`'KMDL'`,
   packed little-endian) before touching the buffer further -- right
   after flashing this runtime, that region is unrelated/erased flash
   content, and it should say so rather than crash.
3. Calls `nncase::runtime::interpreter` (this SDK's vendored nncase C++
   runtime, `lib/nncase/v1/`) directly -- **not** the simplified
   `kpu_load_kmodel`/`kpu_run_kmodel` C wrapper (`lib/nncase/nncase.cpp`)
   most K210 example code uses. The C wrapper doesn't expose input
   shape/dtype queries; going straight to `interp.input_shape(0)`/
   `input_desc(0).datatype` (the same calls `nncase_v1.cpp`'s own
   implementation makes internally) is what lets this runtime size and
   zero its input tensor(s) from whatever model is actually loaded,
   instead of a size hardcoded at build time for one specific model --
   the same "runtime introspects the model" property
   `onnx-cardputer-flash`'s TFLite Micro firmware has via
   `interpreter->input(0)->dims`.
4. Runs one inference over the zeroed input(s) and prints each tensor's
   shape/dtype plus the output byte count. Proves the
   flash-read + interpreter path executes a real model end to end; says
   nothing about accuracy (needs real sensor data, wired up per model)
   or about the KPU hardware specifically -- see the note in
   `onnx-k210-flash/README.md`'s "Model conversion" section about what
   nncase's `Simulator` (used to test the *compiler* side) does and
   doesn't prove, which applies here too, in reverse: this proves the
   *device* runtime loads and runs a model, not that its numerical
   output matches the PC-side `Simulator`'s.

## Getting the toolchain and SDK

Neither is vendored into this repo (matches this repo's own stance on
`onnx2tf`/`nncase`/PlatformIO -- external toolchains stay external):

```sh
# RISC-V toolchain (prebuilt, ~20MB)
curl -LO https://github.com/kendryte/kendryte-gnu-toolchain/releases/download/v8.2.0-20190409/kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz
tar xf kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz

# SDK
git clone https://github.com/kendryte/kendryte-standalone-sdk.git
```

## Building it yourself

```sh
mkdir -p kendryte-standalone-sdk/src/onnx_k210_runtime
cp src/main.cpp src/w25qxx.c src/w25qxx.h src/project.cmake \
   kendryte-standalone-sdk/src/onnx_k210_runtime/

cd kendryte-standalone-sdk
mkdir build && cd build
cmake .. -DPROJ=onnx_k210_runtime -DTOOLCHAIN=/path/to/kendryte-toolchain/bin
ninja   # or `make`, if the SDK picked Unix Makefiles instead of Ninja

# onnx_k210_runtime.bin is the flashable image -- no separate merge step
# (unlike ESP32-S3): objcopy already produces one complete raw image.
```

**Two real build issues hit and fixed, kept here so the next person
doesn't rediscover them:**

- **`w25qxx.c`/`.h` aren't part of the base SDK.** They live in the
  separate `kendryte-standalone-demo` repo (its `kpu`/`flash_w25qxx*`
  example directories) -- vendored here (`src/w25qxx.c`/`.h`, Apache 2.0,
  Copyright 2018 Canaan Inc., per the file's own header) rather than
  requiring a second repo clone. They have no `extern "C"` guard, so
  `main.cpp` wraps its own `#include "w25qxx.h"` in `extern "C" { ... }`
  instead of editing the vendored file.
- **`lib/nncase/v1`'s headers need an include root the SDK's own
  top-level `CMakeLists.txt` (marked "DO NOT MODIFY") never adds.** Its
  `header_directories()` macro globs every `.h` file and adds each one's
  *own* containing directory as an include path -- never the actual
  `lib/nncase/v1/include` parent, which `nncase/runtime/*.h`'s internal
  `<nncase/runtime/...>` angle-bracket includes need. Worse:
  `lib/nncase/v0/` has its own same-named `runtime/interpreter.h`, so a
  naive attempt to work around this by dropping the `nncase/` prefix
  compiles against the *wrong* interpreter silently rather than failing.
  Fixed via `src/onnx_k210_runtime/project.cmake` -- the SDK's own
  documented per-project hook (`cmake/executable.cmake` conditionally
  includes it if present), not a modification to any protected file --
  adding the missing include root plus the vendored `gsl-lite`/
  `mpark-variant`/`nlohmann_json`/`xtl` third-party headers those nncase
  headers transitively need (`third_party/*/include`, already vendored
  in the SDK, just not globally exposed the same way).

## Flash layout

| Region | Offset | Size | Purpose |
|---|---|---|---|
| Firmware (this runtime) | `0x0` | ~1MB (as built) | this compiled image |
| Model | `0x00C00000` | up to `2MB` (reserved) | swappable -- flash a `.kmodel` here |

`0x00C00000` isn't an offset invented for this tool -- it's Kendryte's own
convention (`kendryte-standalone-demo/kpu/kfpkg/flash-list.json`'s model
address), reused for consistency with the rest of the K210 ecosystem
(kflash_gui's `.kfpkg` multi-file flashing already expects models around
this address in several official demos).

## Flashing a model without re-erasing (verified safe on real hardware)

`k210_isp.mjs`'s `flashFirmware()` calls `flashErase()` by default, and
`FLASH_ERASE` (0xd3) has **no address or range parameter at the protocol
level** -- confirmed against kflash.py's own source (`onnx-k210-flash`'s
own README covers this): it always erases the *entire* chip. That means
flashing the runtime, then later flashing a model with erase left on,
would erase the runtime out from under itself before writing the model.

The Web UI exposes a **"skip erase"** checkbox (off by default) and a
**flash address** field for exactly this reason. Whether it's actually
*safe* to skip erase when writing just the model -- i.e. whether the
flash-mode stub's own `FLASH_WRITE` (0xd4) implementation erases the
sectors it's about to program before writing them, the way some embedded
NOR-flash write helpers do as a convenience -- used to be unverified here.
It's now been answered for real, if by accident: see `../../README.md`'s
"Testing against real hardware", finding #4. Summary: a real
firmware-corrupting write to `0x0` (a different mistake, kflash's own
`-A`/`--addr` not applying to a plain file write -- finding #3), followed
by a correct re-flash of the real firmware to that same now-corrupted
region, both **without** an explicit chip erase, produced a byte-correct,
working firmware. NOR flash writes can only clear bits, never set them, so
that result is only possible if `FLASH_WRITE` auto-erases the sectors it
programs. Separately, writing a model to `0x00C00000` left the firmware at
`0x0` completely intact and still booting correctly. Both point the same
way: **skip-erase is safe**, at least on the real Sipeed Maix Amigo this
was tested against -- not proven for every K210 board/flash-chip
combination, but no longer a total unknown either.

## Using it

1. Flash `prebuilt/onnx-k210-runtime.bin` at `0x0` with `onnx-k210-flash`'s
   Web Serial flasher (`../../web/index.html`), erase **on** (default).
2. Convert a model (`../../scripts/onnx_to_kmodel.py`) and flash the
   resulting `.kmodel` file's bytes at `0x00C00000` -- with erase **on**
   too, until "Flashing a model without re-erasing" above has a verified
   answer for your hardware. That erases the runtime along with it, so
   re-flash `onnx-k210-runtime.bin` at `0x0` again afterward, same
   Connect session, erase off is fine for that second write since the
   chip is already freshly erased from step 2's erase.
3. Reset the board -- it prints what it loaded over UART.
