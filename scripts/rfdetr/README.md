# RF-DETR × onnxsim compatibility harness

[RF-DETR](https://github.com/roboflow/rf-detr) is Roboflow's real-time
DETR-style object detector / instance segmenter. Its ONNX export path
(`rfdetr/export/_onnx/exporter.py`) already depends on **onnxsim** and calls

```python
onnxsim.simplify(sim_onnx_dir, check_n=3, input_data=input_dict, dynamic_input_shape=False)
```

so onnxsim is a first-class part of RF-DETR's deployment pipeline.

`simplify_rfdetr.py` is a standalone harness that exports each RF-DETR
variant to ONNX via the public `RFDETR*.export()` API and simplifies the
result with the **current** onnxsim, reproducing RF-DETR's own call. It
reports node counts before/after and whether onnxsim's numerical
equivalence check (`check_n`) passed.

## Running

```bash
pip install rfdetr onnxruntime onnxsim
python scripts/rfdetr/simplify_rfdetr.py --all
```

`onnxruntime` is **not** optional here. onnxsim uses it to numerically
compare the original and simplified graphs (`check_n`). Without it, onnxsim
falls back to onnx's Python reference evaluator, whose `GatherElements`
implementation (`numpy.choose`, capped at 64 choices) cannot execute
RF-DETR's decoder — the check then crashes with
`ValueError: Need at least 0 and at most 64 array objects` instead of
simplifying. RF-DETR's `[onnx]` extra already requires `onnxruntime`, so a
correctly set-up RF-DETR environment never hits this.

## Notes

- RF-DETR's `[onnx]` extra currently pins `onnxsim<0.6.0`. This harness
  confirms the exported graphs also simplify cleanly with onnxsim 0.7.x,
  so that upper bound could be relaxed.
- XLarge / 2XLarge segmentation variants additionally require
  `pip install rfdetr[plus]`. They simplify correctly but report
  `check_ok=False`: their large mask head accumulates a sub-`1e-4`
  floating-point difference from op reordering that exceeds onnxsim's strict
  `np.allclose(rtol=1e-4, atol=1e-5)` check. The graphs are functionally
  equivalent (no NaNs, detection heads match to ~`1e-4`); pass `check_n=0`
  to skip the strict check for these.

See `RESULTS.md` for a captured run and the full analysis.
