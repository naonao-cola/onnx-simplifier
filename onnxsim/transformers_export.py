"""Export a Hugging Face ``transformers`` model straight to a simplified
ONNX deployment directory.

onnxsim has no PyTorch tracing code of its own, and does not need any: turning
a ``transformers`` model into an ONNX graph is exactly what Hugging Face's own
``optimum`` package already does, for every architecture that has an
``optimum.exporters.onnx`` ``OnnxConfig`` (hundreds of model types, including
the split multi-file encoder/decoder-with-past shape autoregressive
generation needs -- see ``tests/test_optimum_export_deploy.py``). That export
is deliberately plain -- no runtime-specific op fusion -- so there is real
simplification left on the table for onnxsim's own pipeline to find.

This is a different tool for a different job than ONNX Runtime GenAI's own
model builder (``onnxruntime_genai.models.builder``): that one covers a fixed,
curated list of decoder-only causal-LM architectures, and its output is
already fused/quantized into ORT-specific ops (e.g. ``com.microsoft::
MatMulNBits``, ``GroupQueryAttention``) meant to be consumed directly by ORT
GenAI's own generate() loop -- there is little left for a generic simplifier
to do to it, and it does not cover encoder-only, seq2seq, vision, or audio
architectures at all. ``optimum``'s export is the right shape for *this*
job instead: a plain graph, for any architecture with an ``OnnxConfig``,
handed to onnxsim to clean up.

:func:`export_transformers_model` wraps exactly the manual
export-then-simplify-then-copy-the-rest recipe
``tests/test_optimum_export_deploy.py`` exercises by hand, as a reusable
onnxsim entry point.
"""

import glob
import os
from typing import Dict, Optional

import onnx
from google.protobuf.message import EncodeError

from onnxsim.onnx_simplifier import simplify


def export_transformers_model(
    model_id: str,
    output_dir: str,
    task: str = "auto",
    no_post_process: bool = True,
    check_n: int = 0,
    export_kwargs: Optional[Dict] = None,
    simplify_kwargs: Optional[Dict] = None,
) -> Dict[str, bool]:
    """Export ``model_id`` (a Hugging Face Hub id or local model directory) to
    ONNX via ``optimum.exporters.onnx.main_export``, then simplify every
    ``.onnx`` file it produces, in place, inside ``output_dir``.

    Needs the optional ``torch``, ``transformers``, and ``optimum`` (with the
    ``optimum-onnx`` distribution installed for ``optimum.exporters.onnx``)
    packages -- heavy, and unrelated to onnxsim's own ONNX-to-ONNX pipeline,
    so they are not normal onnxsim dependencies
    (``pip install onnxsim[transformers]``).

    :param model_id: Hugging Face Hub model id or local model directory
    :param output_dir: directory to export into. Also where the simplified
            files end up: each exported ``.onnx`` file is overwritten in
            place with its simplified version. Non-``.onnx`` files (tokenizer,
            config, ...) are left untouched, so the directory stays
            deployable exactly like a plain ``optimum`` export.
    :param task: the export task, e.g. ``"text-generation-with-past"`` or
            ``"text2text-generation-with-past"``; ``"auto"`` (the default)
            lets ``optimum`` infer it from the model's config.
    :param no_post_process: keep a multi-file encoder/decoder(-with-past)
            export split rather than merged into a single graph with an
            ``If``-node branch switch. Defaults to ``True``: as of this
            writing, simplifying a merged decoder produces a model that
            fails at runtime (see ``tests/test_optimum_export_deploy.py``'s
            docstring for the specific failure) -- the split shape simplifies
            and reloads correctly, and ``optimum``'s own runtime classes
            (e.g. ``ORTModelForSeq2SeqLM``) fall back to it automatically
            when no merged file is present.
    :param check_n: forwarded to :func:`onnxsim.simplify` for every exported
            graph -- how many random-input runs to check the simplified
            model against the freshly exported one for numerical equivalence.
    :param export_kwargs: extra keyword arguments forwarded to
            ``optimum.exporters.onnx.main_export`` (e.g. ``opset``,
            ``device``, ``fp16``, ``trust_remote_code``).
    :param simplify_kwargs: extra keyword arguments forwarded to
            :func:`onnxsim.simplify` for every exported graph.
    :returns: ``{filename: check_ok}`` for every ``.onnx`` file exported,
            where ``check_ok`` is that file's :func:`onnxsim.simplify`
            numerical-equivalence check result (always ``True`` when
            ``check_n == 0``, since no check is performed).
    """
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as e:
        raise ImportError(
            "export_transformers_model needs the optional 'torch', "
            "'transformers', and 'optimum' (with the 'optimum-onnx' "
            "distribution) packages: pip install onnxsim[transformers]"
        ) from e

    main_export(
        model_id,
        output=output_dir,
        task=task,
        no_post_process=no_post_process,
        **(export_kwargs or {}),
    )

    results = {}
    for src in sorted(glob.glob(os.path.join(output_dir, "*.onnx"))):
        model_opt, check_ok = simplify(src, check_n=check_n, **(simplify_kwargs or {}))
        try:
            onnx.save(model_opt, src)
        except (ValueError, EncodeError):
            # Real transformers models routinely exceed onnx.save's 2GB inline
            # limit; fall back to external data next to the model, matching
            # the CLI's own --save-as-external-data fallback (onnx_simplifier.py).
            external_data_path = os.path.basename(src) + ".data"
            if os.path.exists(os.path.join(output_dir, external_data_path)):
                os.remove(os.path.join(output_dir, external_data_path))
            onnx.save(
                model_opt,
                src,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path,
            )
        results[os.path.basename(src)] = check_ok
    return results
