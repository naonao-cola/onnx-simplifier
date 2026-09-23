"""Generate the real test data for client_main.c's boxhead-GEMM real-kernel test:
m=64,k=12544,n=512 -- k is the real fc6 box-head reduction depth (7*7*256, RoiAlign's flattened
crop); m/n are a real-shape-consistent slice of the real m~1000,n=1024 shape, chosen so the packed
weight buffer (k*n*4 bytes) fits under the ~32MB/buffer RPC transfer wall this project's own
native_transport work already found (the full real weight, 12544*1024*4 ~= 49MB, doesn't).
Matches hex_boxhead_gemm_kernel.py's own pack_b() layout and main()'s random seed exactly (same
convention gen_gemm_test_data.py established for hex_gemm_kernel.py), so this is byte-for-byte the
same input `hex_boxhead_gemm_kernel.py --m 64 --k 12544 --n 512` already verified correct under
qemu before mini_rpc_impl.c's hex_boxhead_gemm() function was pasted in from its output.

Run this, then `bash build.sh` from the same directory (it needs a small addition to pick up
boxhead_a.bin/boxhead_bp.bin the same way it already does for gemm_a.bin/gemm_bp.bin)."""
import numpy as np


def pack_b(b: np.ndarray, n_tile: int = 32) -> np.ndarray:
    k, n = b.shape
    return b.reshape(k, n // n_tile, n_tile).transpose(1, 0, 2).copy()


if __name__ == "__main__":
    m, k, n = 64, 12544, 512
    rng = np.random.default_rng(7)
    a_np = rng.normal(0, 1, (m, k)).astype(np.float32)
    b_np = rng.normal(0, 1, (k, n)).astype(np.float32)
    bp_np = pack_b(b_np)
    ref = a_np.astype(np.float64) @ b_np.astype(np.float64)

    a_np.tofile("boxhead_a.bin")
    bp_np.tofile("boxhead_bp.bin")
    ref.astype(np.float32).tofile("boxhead_ref.bin")
    print(f"wrote boxhead_a.bin ({a_np.nbytes}), boxhead_bp.bin ({bp_np.nbytes}), "
          f"boxhead_ref.bin ({ref.astype(np.float32).nbytes}) for m={m} k={k} n={n}")
