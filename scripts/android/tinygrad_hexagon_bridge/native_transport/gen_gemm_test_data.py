"""Generate the real test data for client_main.c's real-kernel test: cin=64,cout=256,m=54400
(the flagship Mask R-CNN backbone shape, hex_gemm_kernel.py's own default), matching
hex_gemm_kernel.py's own pack_b() layout and main()'s random seed exactly, so this is byte-for-
byte the same input hex_gemm_kernel.py --cin 64 --cout 256 --m 54400 already verified correct
under qemu before mini_rpc_impl.c's hex_gemm() function was pasted in from its output.

Run this, then `bash build.sh` from the same directory (it picks up gemm_a.bin/gemm_bp.bin
automatically if present) to exercise the real kernel end to end."""
import numpy as np


def pack_b(b: np.ndarray, n_tile: int = 32, k_sub: int = 4) -> np.ndarray:
    K, N = b.shape
    return (
        b.reshape(K // k_sub, k_sub, N // n_tile, n_tile)
        .transpose(2, 0, 3, 1)
        .reshape(N // n_tile, K // k_sub, n_tile * k_sub)
    )


if __name__ == "__main__":
    cin, cout, m = 64, 256, 54400
    rng = np.random.default_rng(5)
    a_np = rng.integers(0, 100, (m, cin)).astype(np.uint8)
    b_np = rng.integers(0, 100, (cin, cout)).astype(np.uint8)
    bp_np = pack_b(b_np)
    ref = a_np.astype(np.int64) @ b_np.astype(np.int64)

    a_np.tofile("gemm_a.bin")
    bp_np.tofile("gemm_bp.bin")
    ref.astype(np.int32).tofile("gemm_ref.bin")
    print(f"wrote gemm_a.bin ({a_np.nbytes}), gemm_bp.bin ({bp_np.nbytes}), "
          f"gemm_ref.bin ({ref.astype(np.int32).nbytes}) for cin={cin} cout={cout} m={m}")
