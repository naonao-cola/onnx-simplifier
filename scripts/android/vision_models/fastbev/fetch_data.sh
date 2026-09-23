#!/bin/sh
# Download the official Fast-BEV M0 and Fast-BEV++ R50 checkpoints (Google Drive, via gdown) and
# reuse ../bevformer_tiny/fetch_data.sh's camera-only nuScenes-mini slice. Nothing is committed.
#   Fast-BEV   Sense-GVT/Fast-BEV README "download" (work_dirs/.../fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth)
#   Fast-BEV++ ymlab/advanced-fastbev README "Model Zoo" fastbev-r50-cbgs (epoch_20_ema.pth)
# usage: fetch_data.sh [dir]   (default ~/.cache/fastbev; needs `pip install gdown`)
set -eu
D=${1:-$HOME/.cache/fastbev}
mkdir -p "$D/ckpt"
get() {  # id file sha256
  if ! echo "$3  $D/ckpt/$2" | sha256sum -c --status 2>/dev/null; then
    python3 -m gdown -q "https://drive.google.com/uc?id=$1" -O "$D/ckpt/$2"
    echo "$3  $D/ckpt/$2" | sha256sum -c
  fi
}
get 1_L3y6LMV9BAFJw0XaZRTNo-kDlhgTS17 m0_epoch_20.pth db2326190537a777fdcf6c4f76950510b1219fc0ecc4beb1c9375fc99b377c65
get 1q49_7aR7EWWOCAUK8iE5jfj1JdtL9gJy fbpp_r50_cbgs_epoch_20_ema.pth 4488f70ce58d2313b21a5cf26fa4dee96599d4683a4bb225814da6f4d90ef4f2
"$(dirname "$0")/../bevformer_tiny/fetch_data.sh" "${NUSCENES_DIR:-$HOME/.cache/onnxsim-bevformer}" | grep nuscenes
