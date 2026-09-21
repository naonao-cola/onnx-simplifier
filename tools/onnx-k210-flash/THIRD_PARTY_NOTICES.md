# Third-party notices

## kflash.py

`web/k210_isp.mjs` is a from-scratch JavaScript port of the ISP flashing
protocol implemented by
[kendryte/kflash.py](https://github.com/kendryte/kflash.py) -- every
constant, packet layout, and control sequence was read out of that
project's `kflash.py` and re-implemented in JavaScript for Web Serial; no
code was copied.

`web/isp_stub.bin` **is** copied, byte-for-byte: it's kflash.py's own
`ISP_PROG` constant (an embedded, zlib-compressed binary), decompressed.
That binary is the "flash mode" firmware the K210's mask-ROM ISP loads into
SRAM and boots into partway through flashing -- the mask ROM itself can
only read/write SRAM, not the SPI flash, so this stub is what actually
implements flash erase/write. There is no source for it beyond the
compiled bytes kflash.py ships, so vendoring those bytes is the only way to
reuse it.

kflash.py's license (as of the commit this was ported from):

```
MIT License

Copyright (c) 2019 Kendryte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## kendryte-standalone-demo (w25qxx.c / w25qxx.h)

`firmware/runtime/src/w25qxx.c` and `.h` are copied, as-is, from
[kendryte/kendryte-standalone-demo](https://github.com/kendryte/kendryte-standalone-demo)'s
`kpu/` example directory -- the SPI-NOR-flash driver
`firmware/runtime/src/main.cpp` uses to read a kmodel off flash at boot.
It isn't part of the base `kendryte-standalone-sdk` (only this separate
demo repo), and isn't otherwise available as source. Per the files' own
header comment:

```
Copyright 2018 Canaan Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## kendryte-standalone-sdk

`firmware/runtime/prebuilt/onnx-k210-runtime.bin` is built against
[kendryte/kendryte-standalone-sdk](https://github.com/kendryte/kendryte-standalone-sdk)
(Apache License 2.0) -- its `lib/` (drivers, the vendored `nncase` v0/v1
runtime including prebuilt `lib/nncase/v1/lib/*.a`, and `third_party/`
gsl-lite/mpark-variant/nlohmann_json/xtl) links into that binary. The SDK
itself isn't vendored into this repo (see `firmware/runtime/README.md`
for how to fetch it to rebuild); this note covers what ends up compiled
into the committed `.bin`.

## MaixPy-v1 and OpenMV (camera demo)

`firmware/camera_demo/src/m5stickv.c` is not a copy of, but is derived from
reading, [sipeed/MaixPy-v1](https://github.com/sipeed/MaixPy-v1) (released
under MIT or Apache 2.0, at the reader's choice): the M5StickV pin map, the
AXP192 register sequence (`components/boards/m5stick`), the ST7789 command
sequence, panel offsets and M5StickV register tweaks (`components/drivers/lcd`,
`projects/maixpy_m5stickv/builtin_py/_boot.py`), the DVP/interrupt setup
(`components/micropython/port/src/omv/sensor.c`) and the
camera-through-hardware-I2C2 arrangement (`.../omv/cambus.c`).

The OV7740 register table in that file (`ov7740_regs`) is OpenMV's default
OV7740 configuration as shipped in MaixPy
(`components/micropython/port/src/omv/ov7740.c`), MIT-licensed:

```
This file is part of the OpenMV project.
Copyright (c) 2013/2014 Ibrahim Abdelkader <i.abdalkader@gmail.com>
This work is licensed under the MIT license, see the file LICENSE for details.
```
