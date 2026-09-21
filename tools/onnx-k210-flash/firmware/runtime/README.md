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

**Built from source, flashed and run end to end on a real M5StickV**
(2026-09-21): firmware at `0x0`, a `.kmodel` at `0x00C00000`, and the runtime
loads it and runs an inference. Earlier (Sipeed Maix Amigo) this runtime
booted and detected a blank model region correctly but hung in
`interpreter::load_model()` given a real kmodel; that turned out to be two
separate bugs, both fixed here:

1. **Missing KPU clock/DMA setup (the hang).** `main()` only configured PLL0.
   The k210 runtime module touches KPU registers while loading, and with PLL1
   (the KPU clock source) unconfigured and the AI clock gate closed that access
   never returns. `main()` now sets PLL1/PLL2, calls `dmac_init()` and
   `sysctl_clock_enable(SYSCTL_CLOCK_AI)`, as kendryte's own kpu demos do. (All
   four were applied together; which of them is strictly necessary wasn't
   isolated.)
2. **nncase compiler/runtime version mismatch (failure at `run()`).** The SDK
   vendors nncase runtime **1.0.0** (2021) in `lib/nncase/v1`, while
   `../../scripts/onnx_to_kmodel.py` compiles with **nncase 1.9.0**. With the
   mismatch, `load_model()` succeeds but `run()` fails with
   `runtime_module.cpp:65 (shape_reg) id < shape_regs_.size()` /
   "Result too large". `build.sh` swaps in the matching runtime from the
   [kendryte/nncase v1.9.0 release](https://github.com/kendryte/nncase/releases/tag/v1.9.0)
   (`nncaseruntime-riscv64-none-k210.zip`).

**Verified output.** A 2-conv test net (`1x3x32x32` float32 in, `1x10` out,
5240-byte kmodel) fed the ramp `x[j] = ((j*7)%256)/255 - 0.5`, compared with
onnxruntime on the same ramp: correlation 0.98, same argmax/argmin, max
absolute error 0.45 -- the expected size of uint8 PTQ error (calibrated on
random data, ramp input outside that distribution), not bit-exactness.
The runtime doesn't report whether ops ran on the KPU or fell back to CPU.

Things learned while testing on the M5StickV (kept because each cost time):

- **Don't test with `kflash -s` (SRAM boot).** The ISP stub leaves state behind
  that makes this runtime's DMA flash read hang -- it *looks* like the same
  `load_model()` hang but happens earlier, before the model is even read.
  Real flash boots don't have it.
- **Opening the serial port resets the M5StickV** (DTR/RTS are wired to reset)
  into whatever is in flash. With pyserial, set `dtr=False; rts=False` before
  `open()` to avoid it; the runtime's banner is printed within milliseconds of
  boot, so read the port promptly.
- **kflash flags on M5StickV:** `-B goE -b 115200`. `-b 1500000` drops the port
  right after the ISP stub boots.
- **A `.kfpkg`'s `address` must be a string** (`"0x00C00000"`); kflash calls
  `int(x, 0)` on it and a JSON number fails with
  `int() can't convert non-string with explicit base`. Set
  `"sha256Prefix": false` for a model (`true` prepends a length+SHA header, and
  the runtime expects `KMDL` at the very start).

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
4. Runs one inference (float32 inputs get a deterministic ramp, others zeros)
   and prints each tensor's shape/dtype, the output byte count, and float32
   output values. Proves the flash-read + interpreter path executes a real
   model end to end, and the printed values can be checked against
   onnxruntime on the host fed the same ramp (see "Status"). That's a
   sanity check with expected uint8-PTQ error, not real sensor data, and it
   says nothing about whether ops ran on the KPU or fell back to CPU.

## Building it yourself

```sh
./build.sh [WORKDIR]     # default WORKDIR: ./build-work (git-ignored)
# -> WORKDIR/onnx-k210-runtime.bin, same as prebuilt/onnx-k210-runtime.bin
```

`build.sh` downloads the Kendryte RISC-V toolchain (`kendryte-gnu-toolchain`
8.2.0) and `kendryte-standalone-sdk` into WORKDIR (neither is vendored -- this
repo's own stance on external toolchains), swaps the SDK's nncase runtime for
1.9.0 (see "Status" above; `sdk-patch/nncase_v1_CMakeLists.txt` is the
replacement `lib/nncase/v1/CMakeLists.txt` -- 1.9.0's cmake package declares an
imported target named `kendryte` that collides with the SDK's own, so the `.a`
files are linked directly), copies `src/` into the SDK as project
`onnx_k210_runtime`, and builds. CMake 4.x needs
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for the SDK's old
`cmake_minimum_required`; the script passes it. The SDK's CMake also resets
`CMAKE_C/CXX_FLAGS`, so extra `-D` defines have to go through `project.cmake`,
not the command line.

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
