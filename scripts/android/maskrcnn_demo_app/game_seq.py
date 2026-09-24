"""The game-upscaling mode's replay sequence: the first N frames of Arm's NSS test sequence (Bistro,
960x540 renders; Arm/neural-graphics-dataset, Arm AI Model Community License -- see README) as one
flat file the app memory-maps, plus the license text next to it.

  python game_seq.py [--frames 48] [--out ~/.cache/arm-nss/game]      (needs ../vision_models/nss/nss.py fetch)

game_seq.bin: int32 header (magic 'NSSQ', frames, H, W), then per frame: 32 float32 scalars (jitter y, x;
depth params x 4; view-projection 4 x 4 row-major; 10 unused), colour RGBA float16 H x W x 4 (the linear
render, alpha 0), motion float16 2 x H x W (the dataset's motion_lr: (row, col) pixels to the previous
frame, as NSS consumes it), depth float32 H x W.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

NSS = Path.home() / ".cache/arm-nss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--out", type=Path, default=NSS / "game")
    a = ap.parse_args()
    import safetensors

    a.out.mkdir(parents=True, exist_ok=True)
    with (
        safetensors.safe_open(NSS / "test_sample.safetensors", framework="numpy") as f,
        open(a.out / "game_seq.bin", "wb") as o,
    ):
        _, _, h, w = f.get_slice("colour_linear").get_shape()
        o.write(np.array([0x5153534E, a.frames, h, w], np.int32).tobytes())  # 'NSSQ'
        for t in range(a.frames):
            sc = np.zeros(32, np.float32)
            sc[0:2] = f.get_slice("jitter")[t].ravel()[:2]
            sc[2:6] = f.get_slice("depth_params")[t].ravel()
            sc[6:22] = f.get_slice("viewProj")[t].reshape(16)
            o.write(sc.tobytes())
            c = f.get_slice("colour_linear")[t].astype(np.float16)  # 3 x H x W
            rgba = np.zeros((h, w, 4), np.float16)
            rgba[..., :3] = c.transpose(1, 2, 0)
            o.write(rgba.tobytes())
            o.write(f.get_slice("motion_lr")[t].astype(np.float16).tobytes())
            o.write(f.get_slice("depth")[t].astype(np.float32).tobytes())
    shutil.copy(
        NSS / "LICENSE_Arm_AI_Model_Community.pdf",
        a.out / "LICENSE_Arm_AI_Model_Community.pdf",
    )
    print(a.out / "game_seq.bin", (a.out / "game_seq.bin").stat().st_size >> 20, "MiB")


if __name__ == "__main__":
    main()
