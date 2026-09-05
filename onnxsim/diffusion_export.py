"""Export a Hugging Face ``diffusers`` pipeline (Stable Diffusion, SDXL, ...)
straight to a simplified ONNX deployment directory.

A diffusion pipeline is not one model but several -- typically a text
encoder, a UNet (or transformer) denoiser, and a VAE encoder/decoder, each
with its own graph -- so, like :func:`onnxsim.export_transformers_model`,
there is no PyTorch tracing code of onnxsim's own here either. Turning a
``diffusers`` pipeline into that set of ONNX graphs is exactly what
``optimum.exporters.onnx.main_export`` already does: it detects a
``model_index.json``-style diffusers pipeline the same way
``optimum-cli export onnx`` does, and exports every sub-model into its own
``<component>/model.onnx`` (``text_encoder/model.onnx``, ``unet/model.onnx``,
``vae_encoder/model.onnx``, ``vae_decoder/model.onnx``, plus e.g.
``text_encoder_2/model.onnx`` for SDXL) alongside the non-ONNX pipeline
assets (``model_index.json``, ``scheduler/``, ``tokenizer/``, ...). That
export is plain, un-fused ONNX, so -- exactly as for a transformers export --
there is real simplification left on the table for onnxsim's own pipeline to
find.

:func:`export_diffusion_model` wraps that export-then-simplify-every-graph
recipe as one call, the diffusion counterpart of
:func:`onnxsim.export_transformers_model`. The two are kept as separate
entry points rather than one, because the on-disk shape they simplify is
different: a transformers export's graphs sit flat in ``output_dir``, while a
diffusion export nests each component's ``model.onnx`` inside its own
subdirectory.
"""

import glob
import os
from typing import Dict, Optional

from onnxsim.onnx_simplifier import simplify
from onnxsim.transformers_export import _save


def export_diffusion_model(
    model_id: str,
    output_dir: str,
    task: str = "auto",
    check_n: int = 0,
    save_as_external_data: bool = True,
    export_kwargs: Optional[Dict] = None,
    simplify_kwargs: Optional[Dict] = None,
) -> Dict[str, bool]:
    """Export ``model_id`` (a Hugging Face Hub id or local ``diffusers``
    pipeline directory) to ONNX via ``optimum.exporters.onnx.main_export``,
    then simplify every ``.onnx`` file it produces, in place, inside
    ``output_dir``.

    Needs the optional ``torch``, ``diffusers``, and ``optimum`` (with the
    ``optimum-onnx`` distribution installed for ``optimum.exporters.onnx``)
    packages -- heavy, and unrelated to onnxsim's own ONNX-to-ONNX pipeline,
    so they are not normal onnxsim dependencies
    (``pip install onnxsim[diffusion]``).

    :param model_id: Hugging Face Hub model id or local ``diffusers``
            pipeline directory (a ``model_index.json`` plus per-component
            subdirectories).
    :param output_dir: directory to export into. Also where the simplified
            files end up: each exported ``<component>/model.onnx`` is
            overwritten in place with its simplified version. Non-``.onnx``
            files (``model_index.json``, ``scheduler/``, ``tokenizer/``, the
            per-component ``config.json``, ...) are left untouched, so the
            directory stays deployable exactly like a plain ``optimum``
            export (e.g. via ``optimum.onnxruntime.ORTStableDiffusionPipeline
            .from_pretrained``).
    :param task: the export task, e.g. ``"stable-diffusion"`` or
            ``"stable-diffusion-xl"``; ``"auto"`` (the default) lets
            ``optimum`` infer it from the pipeline's ``model_index.json``.
    :param check_n: forwarded to :func:`onnxsim.simplify` for every exported
            graph -- how many random-input runs to check the simplified
            model against the freshly exported one for numerical equivalence.
    :param save_as_external_data: always save every simplified graph with its
            weights in a companion ``<filename>.data`` file, instead of
            inline. On by default here -- unlike the ``onnxsim`` CLI's own
            ``--save-as-external-data``/plain ``onnx.save``, which default
            off and only use external data as a fallback once a graph is too
            large to serialize inline at all (>2GB) -- for the same reason as
            :func:`onnxsim.export_transformers_model`: a real (non-tiny)
            diffusion export's UNet in particular routinely exceeds that
            limit on its own, and every pass in onnxsim's own optimization
            pipeline that touches a graph (shape inference, checker, each
            fixed-point round) copies its inline bytes along with it.
            External data keeps the large tensors on disk instead, so that
            repeated in-memory copying shrinks to metadata (name/offset/
            length) rather than the tensors themselves. Pass ``False`` to
            keep small/tiny pipelines (tests, toy checkpoints) as
            self-contained ``.onnx`` files with no companion ``.data``.
    :param export_kwargs: extra keyword arguments forwarded to
            ``optimum.exporters.onnx.main_export`` (e.g. ``opset``,
            ``device``, ``fp16``).
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify` for every exported graph.
    :returns: ``{relative_path: check_ok}`` for every ``.onnx`` file
            exported (e.g. ``"unet/model.onnx"``), where ``check_ok`` is that
            file's :func:`onnxsim.simplify` numerical-equivalence check
            result (always ``True`` when ``check_n == 0``, since no check is
            performed).
    """
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as e:
        raise ImportError(
            "export_diffusion_model needs the optional 'torch', "
            "'diffusers', and 'optimum' (with the 'optimum-onnx' "
            "distribution) packages: pip install onnxsim[diffusion]"
        ) from e

    main_export(
        model_id,
        output=output_dir,
        task=task,
        **(export_kwargs or {}),
    )

    results = {}
    pattern = os.path.join(output_dir, "**", "*.onnx")
    for src in sorted(glob.glob(pattern, recursive=True)):
        model_opt, check_ok = simplify(src, check_n=check_n, **(simplify_kwargs or {}))
        _save(model_opt, src, force_external_data=save_as_external_data)
        rel = os.path.relpath(src, output_dir).replace(os.sep, "/")
        results[rel] = check_ok
    return results
