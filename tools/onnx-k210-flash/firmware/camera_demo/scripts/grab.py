#!/usr/bin/env python3
"""Pull one frame off the M5StickV camera demo over UART and save it as PNGs.

    grab.py [--port /dev/ttyUSB0] [--out DIR]

Sends 'd' to the running demo, which replies with the DVP's RGB "AI" planes,
the model's float input, and its output (see main.cpp's dump_chunk). Writes:

    camera.png       the frame -- the DVP's planar RGB888 output, which is both
                     what the LCD shows and what the model is fed
    model_out.png    a single-channel float32 model output, normalised
    model_out.npy    the raw output floats (needs numpy)

No third-party dependencies (PNG is written with zlib). The port is opened with
DTR/RTS low: on the M5StickV asserting them resets the board.
"""
import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

W, H = 240, 135
MAGIC = b"\xaa\x55K210DM"


def png(path, w, h, rgb):
    raw = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b""))


def rgb565_to_rgb(pixels):
    out = bytearray()
    for v in pixels:
        out += bytes(((v >> 11) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31))
    return bytes(out)


def read_chunks(port, timeout):
    import serial

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 115200, 0.5
    s.dtr = s.rts = False
    s.open()
    s.reset_input_buffer()
    # Opening the port can still reset the board; wait until the demo is
    # actually running (it prints a stats line every second) before asking.
    buf, t0 = b"", time.time()
    while b" fps " not in buf and time.time() < t0 + 30:
        buf += s.read(4096)
    if b" fps " not in buf:
        raise SystemExit("the camera demo isn't printing stats (not running, or no camera?)")
    s.reset_input_buffer()
    s.write(b"d")
    buf, chunks, deadline = b"", {}, time.time() + timeout
    while time.time() < deadline:
        buf += s.read(65536)
        while True:
            i = buf.find(MAGIC)
            if i < 0 or len(buf) < i + 16:
                break
            kind, n = struct.unpack("<II", buf[i + 8:i + 16])
            if len(buf) < i + 16 + n:
                break
            chunks[kind] = buf[i + 16:i + 16 + n]
            buf = buf[i + 16 + n:]
        if 1 in chunks and (4 in chunks or time.time() > deadline - timeout + 45):
            break
    s.close()
    return chunks


def compare(onnx_path, model_in, model_out, out_shape):
    """Run the .onnx on the device's exact float input; report agreement with the device's output."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    inp = sess.get_inputs()[0]
    x = np.frombuffer(model_in, dtype=np.float32).reshape(inp.shape)
    ref = sess.run(None, {inp.name: x})[0].reshape(-1)
    dev = np.frombuffer(model_out, dtype=np.float32).reshape(-1)
    corr = float(np.corrcoef(ref, dev)[0, 1]) if ref.std() > 0 and dev.std() > 0 else float("nan")
    print("vs onnxruntime: corr %.4f, max|diff| %.4f (ref range %.3f..%.3f, device %.3f..%.3f)"
          % (corr, np.abs(ref - dev).max(), ref.min(), ref.max(), dev.min(), dev.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--out", default=".")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--onnx", help="the model's .onnx: also run it with onnxruntime on the device's exact input and compare")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    chunks = read_chunks(a.port, a.timeout)
    if 1 not in chunks:
        sys.exit("no frame received (is the camera demo running?)")

    p = chunks[1]
    rgb = bytearray(W * H * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = p[:W * H], p[W * H:2 * W * H], p[2 * W * H:]
    png(out / "camera.png", W, H, bytes(rgb))
    print("wrote camera.png")

    if 3 in chunks and 4 in chunks:
        hdr = struct.unpack("<5I", chunks[3])
        shape = hdr[1:1 + hdr[0]]
        vals = struct.unpack(f"<{len(chunks[4]) // 4}f", chunks[4])
        print("model output shape", shape, "min %.4f max %.4f" % (min(vals), max(vals)))
        try:
            import numpy as np

            np.save(out / "model_out.npy", np.array(vals, dtype=np.float32).reshape(shape))
        except ImportError:
            pass
        if len(shape) == 4 and shape[1] == 1:
            oh, ow = shape[2], shape[3]
            mx = max(max(vals), 1e-9)
            g = bytes(min(255, int(v / mx * 255)) for v in vals)
            rgb = b"".join(bytes((v, v, v)) for v in g)
            png(out / "model_out.png", ow, oh, rgb)
            print("wrote model_out.png")
    if 2 in chunks:
        (out / "model_in.f32").write_bytes(chunks[2])
        if a.onnx and 4 in chunks:
            compare(a.onnx, chunks[2], chunks[4], shape)


if __name__ == "__main__":
    main()
