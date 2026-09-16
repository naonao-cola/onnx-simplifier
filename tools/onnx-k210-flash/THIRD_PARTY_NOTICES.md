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
