# Sparse4D v3 on the Snapdragon 8+ Gen 1 (Xiaomi 12S, Hexagon V69)

[Sparse4D v3](https://github.com/HorizonRobotics/Sparse4D) (`sparse4dv3_temporal_r50_1x8_bs6_256x704`,
the only public v3 checkpoint: ResNet-50, 6 cameras at 256x704, 900 anchors of which 600 are carried
between frames, 6 decoder layers) rebuilt in plain PyTorch and run on the phone.

Same data and GT criteria as `../bevformer_tiny` and Fast-BEV: nuScenes-mini scene-0103, its first
6 keyframes chained with the temporal state; a detection matches a GT object if its score is
>= 0.3, the class is the same and the BEV centers are within 2 m (greedy, 190 GT objects).

| file | what |
|---|---|
| `model.py` | the model without mmcv/mmdet3d (`load_official()` loads the checkpoint strictly), the host-side temporal instance bank, the decoder, and DFA twice: upstream's rank-6 code verbatim and a rank-4 rewrite |
| `data.py` | nuScenes-mini keyframes as the upstream test pipeline makes them (reuses bevformer_tiny's data) |
| `validate.py` | fp32 over the scene: GT match, and the rank-4 DFA checked against upstream's at every call |
| `export.py` | ONNX pieces for the HTP, checked against torch on ORT CPU, simplified with onnxsim |

## Reproduce

```sh
C=~/.cache/sparse4d; S="systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0 --working-directory=$PWD"
curl -L -o $C/sparse4dv3_r50.pth https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth
echo "5beed4d4933ca6448d72586b0f8812863574289ff3c4192de71dc9f46a42f0ed  $C/sparse4dv3_r50.pth" | sha256sum -c
../bevformer_tiny/fetch_data.sh        # nuScenes-mini cameras -> ~/.cache/onnxsim-bevformer/nuscenes-mini
$S python3 validate.py --ckpt $C/sparse4dv3_r50.pth --data ~/.cache/onnxsim-bevformer/nuscenes-mini --work $C/work
$S python3 export.py frame --ckpt $C/sparse4dv3_r50.pth --work $C/work
```

## fp32 (host)

| scene-0103, 6 frames chained | GT matched / 190 | predictions | precision |
|---|---|---|---|
| score >= 0.3 (the bevformer_tiny / Fast-BEV criterion) | 71 | 80 | 89% |
| score >= 0.2 | 116 | 184 | 63% |

Per frame (>= 0.3): 6/23, 8/29, 10/30, 12/35, 16/34, 19/39. Frames get better as the temporal
instances accumulate. Sparse4D's final score is the class probability times the sigmoid of its
centerness estimate. So its scores sit lower than BEVFormer's, and at 0.3 it is precise but
conservative. The comparisons below give both thresholds.

The rank-4 DFA (`dfa_rank4`, what the HTP graphs use) against upstream's rank-6 code, on the same
inputs at all 36 calls: max abs 3.8e-6.
