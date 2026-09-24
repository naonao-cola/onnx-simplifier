#!/bin/sh
# Download the official BEVFormer-tiny checkpoint and a camera-only slice of nuScenes-mini.
# Neither needs an account. Nothing here is committed; default location ~/.cache/onnxsim-bevformer.
#
#   checkpoint  bevformer_tiny_epoch_24.pth (401 MB, sha256-pinned), from the BEVFormer README
#   nuScenes    v1.0-mini.tgz is 4.2 GB, but its metadata JSON and all 6 cameras' keyframes sit in
#               the first ~760 MB (tar order: maps, v1.0-mini/, samples/CAM_*, LIDAR, RADAR,
#               sweeps). A range request streams just that prefix; tar then stops at the cut
#               ("Unexpected EOF"), which is expected. ~450 MB on disk. nuScenes is CC BY-NC-SA 4.0.
set -eu
D=${1:-$HOME/.cache/onnxsim-bevformer}
mkdir -p "$D"
CK="$D/bevformer_tiny_epoch_24.pth"
SHA=7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12
if ! echo "$SHA  $CK" | sha256sum -c --status 2>/dev/null; then
  curl -fL -C - -o "$CK" https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
  echo "$SHA  $CK" | sha256sum -c
fi
NS="$D/nuscenes-mini"
if [ ! -f "$NS/v1.0-mini/sample.json" ] || [ "$(ls "$NS/samples/CAM_FRONT" 2>/dev/null | wc -l)" -lt 404 ]; then
  mkdir -p "$NS"
  curl -fsSL -r 0-779999999 https://www.nuscenes.org/data/v1.0-mini.tgz \
    | tar xz -C "$NS" --wildcards 'v1.0-mini/*' 'samples/CAM_*' 2>/dev/null || true
  for c in CAM_FRONT CAM_FRONT_RIGHT CAM_FRONT_LEFT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
    n=$(ls "$NS/samples/$c" | wc -l)
    [ "$n" -ge 404 ] || { echo "nuscenes-mini: only $n $c images, rerun" >&2; exit 1; }
  done
fi
echo "checkpoint: $CK"
echo "nuscenes-mini: $NS"
