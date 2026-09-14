"""Export a `DriveTransformer <https://github.com/Thinklab-SJTU/DriveTransformer>`_
end-to-end autonomous-driving model straight to a simplified ONNX file.

DriveTransformer (ICLR 2025) is a camera-only, streaming/StreamPETR-style
detector -- built with an ``mmdet3d_plugin`` config (``adzoo/drivetransformer/
configs/drivetransformer/drivetransformer_large.py``) on top of an ``mmcv``
that is *not* the real PyPI ``mmcv`` package: DriveTransformer's own
``adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer.py`` does
``from mmcv.models.builder import DETECTORS`` / ``from mmcv.models.detectors.
mvx_two_stage import MVXTwoStageDetector`` -- submodules the real ``mmcv``
has never had (that's mmdet3d/mmdet territory). Like UniAD/VAD/BEVFormer,
DriveTransformer vendors its own merged ``mmcv``+``mmdet``+``mmdet3d`` fork
inside its own repository, installed only by installing DriveTransformer
itself from source (see :func:`export_drivetransformer_model`'s docstring).

DriveTransformer ships no ONNX exporter of its own (neither an official
script nor a known community one, unlike Detectron2/SAM 2 -- see
:mod:`onnxsim.detectron_export`/:mod:`onnxsim.sam2_export`), and its
``forward()`` is dict-in/dict-out with Python-side postprocessing, so this
module builds its own thin tracing wrapper, the same "flatten the dict
interface into plain tensors" idea as those two modules' ``TracingAdapter``/
encoder-decoder split, just with no upstream recipe to mirror. Every method
name, call signature, and stateful-memory detail cited below was read
directly from DriveTransformer's real source (``adzoo/drivetransformer/
mmdet3d_plugin/ours/drivetransformer.py`` and ``team_code/
drivetransformer_b2d_agent.py``, the real-time CARLA/Bench2Drive inference
agent), not guessed from sibling BEVFormer-family repos:

* **The traceable boundary is ``pts_bbox_head.forward()``'s raw tensor
  output, not the detector's own ``forward_test()``.** ``forward_test()``
  calls ``pts_bbox_head.get_bboxes(outs, img_metas)`` afterwards, which does
  NMS-free top-k decoding and ``.cpu()``/``LiDARInstance3DBoxes`` conversion
  -- not ONNX-traceable, and squarely postprocessing rather than the model
  itself. This module instead calls the same two steps ``forward_test()``
  itself calls before that point -- ``model.extract_img_feat(img, img_metas)``
  then ``model.pts_bbox_head(img_feats, img_metas, ego_lcf_feat, ego_fut_cmd,
  ego_his_trajs, lidar2img=..., cam_intrinsic=..., ego_pose=...,
  ego_pose_inv=..., timestamp=..., prev_exists=...)`` -- directly, with an
  unambiguous single-sample (not test-time-augmentation-nested) input shape,
  and returns ``outs``' tensor-valued entries as-is. This is the same
  "raw, un-postprocessed outputs, caller does the rest" choice
  :func:`onnxsim.detectron_export.export_detectron_model` makes with
  ``do_postprocess=False`` and :mod:`onnxsim.qwen_drive_perception_reconstruct`
  documents for its own detection head.
* **Cold-start, no temporal fusion.** DriveTransformer's "streaming"
  design (task queries as historical context) is implemented as plain
  Python-object state on ``pts_bbox_head`` (``agent_memory_embedding``,
  ``map_memory_embedding``, ``ego_memory_embedding``, ... -- not
  ``nn.Module`` buffers, so a trace only ever sees whatever is in them at
  trace time), refreshed by ``pre_update_memory``/``post_update_memory`` and
  cleared by ``reset_memory()``. ``forward_test()`` itself calls
  ``reset_memory()`` and passes ``prev_exists=0`` on a scene's first frame
  (``img_metas[0]['scene_token'] != self.prev_scene_token``). This module
  calls ``reset_memory()`` immediately before the traced call and passes
  ``prev_exists=0`` to match that exact first-frame path -- the same
  "single call, no temporal fusion" simplification
  :mod:`onnxsim.qwen_drive_perception_reconstruct` makes for its own
  BEVFormer-style ``TemporalSelfAttention`` (see that module's docstring),
  just because it is genuinely what DriveTransformer's own first-frame
  inference does, not a shortcut invented here.
* **Synthetic calibration, no real Bench2Drive/CARLA rig needed.** The real
  agent (``team_code/drivetransformer_b2d_agent.py``) hardcodes a fixed
  6-camera Bench2Drive/CARLA ``lidar2img``/``cam_intrinsic`` rig and derives
  ``ego_pose``/``ego_his_trajs``/``ego_fut_cmd``/``ego_lcf_feat`` from live
  GPS/IMU/route-planner state -- none of which matters for producing a
  structurally-correct, simplifiable ONNX graph (the traced ops are the same
  regardless of the numeric calibration content). This module instead builds
  every one of those tensors directly with the right shape -- identity
  matrices for calibration/pose, zeros elsewhere -- the same "some valid
  input, not the real sensor rig" choice
  :func:`onnxsim.detectron_export.export_detectron_model` makes for its own
  default synthetic image.
* **A minimal ``img_metas``.** ``extract_img_feat()`` takes an ``img_metas``
  parameter but never reads it (confirmed from source -- it's unused dead
  weight in that one method). ``pts_bbox_head.forward()`` does read it, but
  only for image/pad shape bookkeeping (its own reference-point/positional-
  encoding math is calibration-tensor-driven, not ``img_metas``-driven, per
  the same source). This module's ``img_metas`` therefore only carries
  ``scene_token``/``img_shape``/``pad_shape`` -- enough to satisfy that
  bookkeeping, not a byte-exact replica of ``CustomFormatBundle3D``'s real
  per-sample metadata dict.

Scope, narrower than a real closed-loop evaluation run: single frame, single
sample, the released ``drivetransformer_large.py`` config's 6-camera
Bench2Drive layout and (by default) its 384x1056 network input resolution --
override ``num_cams``/``image_size`` for a different DriveTransformer
config. ``use_grid_mask`` (off by default in the released config) is left
entirely to the config/checkpoint, like every other architectural choice
this module doesn't touch.
"""

import io
import os
from typing import Dict, Optional, Tuple

import onnx

from onnxsim.onnx_simplifier import simplify
from onnxsim.transformers_export import _save


def _import_plugin(cfg) -> None:
    """Mirrors the exact ``cfg.plugin``/``cfg.plugin_dir`` handling in
    DriveTransformer's own ``adzoo/drivetransformer/train.py`` and
    ``team_code/drivetransformer_b2d_agent.py`` -- both import the config's
    ``mmdet3d_plugin`` package (registering ``DriveTransformer``/
    ``DriveTransformerlHead`` with ``mmcv``'s ``DETECTORS``/``HEADS``
    registries as an import side effect) by dotting together
    ``os.path.dirname(cfg.plugin_dir)``'s path components, the same
    convention BEVFormer/UniAD/VAD configs of this lineage all share.
    """
    if not getattr(cfg, "plugin", False):
        return
    plugin_dir = getattr(cfg, "plugin_dir", None)
    if plugin_dir is None:
        return

    import importlib

    module_dir = os.path.dirname(plugin_dir).split("/")
    module_path = module_dir[0]
    for m in module_dir[1:]:
        module_path = module_path + "." + m
    importlib.import_module(module_path)


def export_drivetransformer_model(
    config_file: str,
    output_path: str,
    checkpoint: Optional[str] = None,
    image_size: Tuple[int, int] = (384, 1056),
    num_cams: int = 6,
    opset_version: int = 17,
    check_n: int = 0,
    save_as_external_data: bool = False,
    simplify_kwargs: Optional[Dict] = None,
) -> bool:
    """Build a DriveTransformer model from ``config_file``, trace a single
    cold-start (no temporal history) forward pass to ONNX, then simplify the
    result and save it to ``output_path``.

    Needs the optional ``torch`` package *and* a real DriveTransformer
    checkout installed from source -- see this module's own docstring for
    why ``mmcv``/``mmdet``/``mmdet3d`` cannot simply be pip-installed for
    this: ``pip install onnxsim[drivetransformer]`` only pulls in torch.
    Install DriveTransformer itself (its own ``docs/INSTALL.md``, in short
    ``git clone https://github.com/Thinklab-SJTU/DriveTransformer.git &&
    cd DriveTransformer && pip install -v -e .``) and run this with that
    checkout importable (e.g. from inside it, or on ``PYTHONPATH``) so
    ``config_file``'s ``adzoo.drivetransformer...`` plugin import resolves.

    :param config_file: path to a DriveTransformer ``mmdet3d_plugin`` config,
            e.g. ``adzoo/drivetransformer/configs/drivetransformer/
            drivetransformer_large.py`` from a DriveTransformer checkout.
    :param output_path: where to save the simplified ``.onnx`` file.
    :param checkpoint: path to a ``.pth`` checkpoint to load into the model
            (loaded the same way ``train.py``/the Bench2Drive agent do, via
            ``mmcv.utils.load_checkpoint``). If ``None`` (the default), no
            checkpoint is loaded at all -- the model traces with its random
            initialization instead, enough to produce a structurally-correct
            ONNX graph with no checkpoint or network call, e.g. for testing.
    :param image_size: ``(height, width)`` of the already-preprocessed
            network input to trace with -- *not* a raw camera frame's size
            (this module skips DriveTransformer's own dataset-loading/
            resize/crop/normalize pipeline entirely and builds the
            post-preprocessing tensor directly, see this module's
            docstring). Defaults to ``drivetransformer_large.py``'s own
            384x1056 deploy resolution; override for a different config.
    :param num_cams: number of camera views, matching
            ``drivetransformer_large.py``/the Bench2Drive agent's own fixed
            6-camera (front/front-left/front-right/back/back-left/
            back-right) rig by default.
    :param opset_version: ONNX opset to trace with.
    :param check_n: forwarded to :func:`onnxsim.simplify` -- how many
            random-input runs to check the simplified model against the
            freshly traced one for numerical equivalence.
    :param save_as_external_data: save the simplified graph's weights in a
            companion ``<output_path>.data`` file instead of inline. Off by
            default, like :func:`onnxsim.export_detectron_model`.
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify`.
    :returns: the numerical-equivalence check result from
            :func:`onnxsim.simplify` (always ``True`` when ``check_n == 0``,
            since no check is performed).
    """
    try:
        import torch
        from mmcv import Config
        from mmcv.models import build_model
        from mmcv.utils import load_checkpoint
    except ImportError as e:
        raise ImportError(
            "export_drivetransformer_model needs the optional 'torch' "
            "package plus a real DriveTransformer checkout installed from "
            "source. 'mmcv'/'mmdet'/'mmdet3d' here are DriveTransformer's "
            "own bundled, merged fork -- its 'from mmcv.models import "
            "build_model' has no counterpart in the real PyPI 'mmcv' "
            "package -- so 'pip install onnxsim[drivetransformer]' only "
            "pulls in torch. Install DriveTransformer itself from source: "
            "git clone https://github.com/Thinklab-SJTU/DriveTransformer.git "
            "&& cd DriveTransformer && pip install -v -e . (see "
            "docs/INSTALL.md), and run this with that checkout importable."
        ) from e

    cfg = Config.fromfile(config_file)
    _import_plugin(cfg)

    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.eval()

    height, width = image_size
    img = torch.randn(1, num_cams, 3, height, width)
    calib = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, num_cams, 1, 1)
    lidar2img = calib.clone()
    cam_intrinsic = calib.clone()
    ego_pose = torch.eye(4).reshape(1, 4, 4)
    ego_pose_inv = ego_pose.clone()
    timestamp = torch.zeros(1)
    prev_exists = torch.zeros(1)
    ego_his_trajs = torch.zeros(1, 2, 2)
    # DriveTransformerlHead.forward() does ego_lcf_feat.squeeze(1)/
    # ego_fut_cmd.squeeze(1) before concatenating them with
    # ego_his_trajs.flatten(-2, -1) (confirmed from source) -- both need
    # that extra, squeeze-away middle axis (a leftover "queue" dimension
    # from the same multi-frame training layout extract_img_feat's own
    # len_queue reshape uses), unlike ego_his_trajs.
    ego_lcf_feat = torch.zeros(1, 1, 9)
    ego_fut_cmd = torch.zeros(1, 1, 140)

    img_metas = [
        {
            "scene_token": "onnxsim_export",
            "img_shape": [(height, width, 3)] * num_cams,
            "pad_shape": [(height, width, 3)] * num_cams,
        }
    ]

    def _call_head(
        img,
        lidar2img,
        cam_intrinsic,
        ego_pose,
        ego_pose_inv,
        timestamp,
        prev_exists,
        ego_his_trajs,
        ego_lcf_feat,
        ego_fut_cmd,
    ):
        img_feats = model.extract_img_feat(img=img, img_metas=img_metas)
        img_feats = img_feats[model.position_level][:, 0]
        return model.pts_bbox_head(
            img_feats,
            img_metas,
            ego_lcf_feat,
            ego_fut_cmd,
            ego_his_trajs,
            lidar2img=lidar2img,
            cam_intrinsic=cam_intrinsic,
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
            timestamp=timestamp,
            prev_exists=prev_exists,
        )

    args = (
        img,
        lidar2img,
        cam_intrinsic,
        ego_pose,
        ego_pose_inv,
        timestamp,
        prev_exists,
        ego_his_trajs,
        ego_lcf_feat,
        ego_fut_cmd,
    )
    input_names = [
        "img",
        "lidar2img",
        "cam_intrinsic",
        "ego_pose",
        "ego_pose_inv",
        "timestamp",
        "prev_exists",
        "ego_his_trajs",
        "ego_lcf_feat",
        "ego_fut_cmd",
    ]

    # One throwaway eager call, purely to discover which of ``outs``' keys
    # are actually tensor-valued (some, e.g. 'ego_fut_preds_fix_dist'/
    # 'ego_traj_cls_scores', are None depending on the head config) so
    # torch.onnx.export gets a fixed output_names list -- reset_memory()
    # again below undoes the memory-state side effect this call causes
    # (see this module's docstring on why the traced call must see a true
    # cold start).
    model.pts_bbox_head.reset_memory()
    with torch.no_grad():
        outs = _call_head(*args)
    output_names = sorted(k for k, v in outs.items() if isinstance(v, torch.Tensor))

    class _Wrapper(torch.nn.Module):
        def forward(self, *args):
            outs = _call_head(*args)
            return tuple(outs[k] for k in output_names)

    model.pts_bbox_head.reset_memory()
    wrapper = _Wrapper()
    wrapper.eval()

    buf = io.BytesIO()
    with torch.no_grad():
        try:
            # Newer torch versions default torch.onnx.export to the
            # dynamo-based exporter -- pin to the classic TorchScript-based
            # tracer explicitly where that knob exists, matching
            # onnxsim.export_detectron_model/export_sam2_model.
            torch.onnx.export(
                wrapper,
                args,
                buf,
                opset_version=opset_version,
                dynamo=False,
                input_names=input_names,
                output_names=output_names,
            )
        except TypeError:
            buf = io.BytesIO()
            torch.onnx.export(
                wrapper,
                args,
                buf,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
            )
    raw_model = onnx.load_from_string(buf.getvalue())

    model_opt, check_ok = simplify(
        raw_model, check_n=check_n, **(simplify_kwargs or {})
    )
    _save(model_opt, output_path, force_external_data=save_as_external_data)
    return check_ok
