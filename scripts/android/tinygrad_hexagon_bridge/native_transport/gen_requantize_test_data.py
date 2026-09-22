"""Generate the real test data for client_main.c's requantize test: n=2,000,000 int32 elements
(a real-scale, RPC-transfer-safe slice of the stem conv's cin=64,H=400,W=544 output -- the full
13.9M-element size would exceed the ~32MB/buffer RPC transfer wall this project's native_transport
work already found), in_scale=0.02 out_scale=0.05 in_zp=0 out_zp=114 -- matching
hex_requantize_kernel.py's own default and the exact multiplier/shift baked into mini_rpc_impl.c's
hex_requantize() (byte-for-byte the same input hex_requantize_kernel.py already verified correct
under qemu before that function was pasted in from its output).

Run this, then `bash build.sh` from the same directory (it picks up requant_a.bin automatically if
present) to exercise the real kernel end to end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from hex_requantize_kernel import requantize_ref, compute_multiplier_shift  # noqa: E402

if __name__ == "__main__":
    n = 2_000_000
    in_scale, out_scale, in_zp, out_zp = 0.02, 0.05, 0, 114
    multiplier, shift = compute_multiplier_shift(in_scale / out_scale)
    assert (multiplier, shift) == (1717986918, -1), (multiplier, shift)

    rng = np.random.default_rng(7)
    a_np = rng.integers(-2_000_000, 2_000_000, n).astype(np.int32)
    ref = requantize_ref(a_np, in_zp, multiplier, shift, out_zp)

    a_np.tofile("requant_a.bin")
    ref.tofile("requant_ref.bin")
    print(f"wrote requant_a.bin ({a_np.nbytes}), requant_ref.bin ({ref.nbytes}) for n={n}")
