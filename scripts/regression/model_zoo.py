"""Reference regression models by short name.

The regression sweep (see ``README.md``) runs onnxsim over a curated set of real
models hosted in the Hugging Face `onnxmodelzoo <https://huggingface.co/onnxmodelzoo>`_
org, listed in ``models.json``. Every place that needs one of those models has so
far repeated the same ``snapshot_download`` + "find the largest ``.onnx``" dance
(``worker.py``, ``x2paddle/worker.py``, …). This module gathers that into one
call so a test or script can grab a model by its short name::

    from model_zoo import fetch_model

    onnx_path = fetch_model("resnet18d_Opset18")   # -> /path/to/resnet18d.onnx

The name is resolved leniently:

* a short name (``"resnet18d_Opset18"``) is looked up in ``models.json`` and, if
  found, mapped to its full repo id;
* a bare name not in ``models.json`` is assumed to live under ``onnxmodelzoo/``;
* an explicit ``owner/repo`` (any org) is used verbatim, so arbitrary Hugging
  Face repos work too.

Downloads go through ``huggingface_hub`` and land in its cache, so repeated calls
for the same model are effectively free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# Default org for bare model names, matching the regression set in models.json.
ORG = "onnxmodelzoo"

# The curated model set. Kept next to this file so it travels with the scripts.
MODELS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")

# What to pull from a repo: the graph plus any external-data / weight blobs it
# references, skipping docs and metadata. Mirrors the regression workers so a
# model fetched here behaves identically to one fetched by the sweep.
ALLOW_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "*.onnx.data",
    "*.data",
    "*.pb",
    "*.bin",
    "*.weight",
    "*.weights",
]


def list_models(models_json: str = MODELS_JSON) -> List[str]:
    """Return the full repo ids of the curated regression set, in file order."""
    with open(models_json) as f:
        data = json.load(f)
    return [m["id"] for m in data["models"]]


def _short(model_id: str) -> str:
    """The bare name of a repo id (``onnxmodelzoo/resnet18d`` -> ``resnet18d``)."""
    return model_id.split("/", 1)[-1]


def resolve(name: str, models_json: str = MODELS_JSON) -> str:
    """Resolve a user-supplied model reference to a full ``owner/repo`` id.

    * ``"owner/repo"`` (any slash) is returned unchanged.
    * a short name present in ``models.json`` maps to its listed id.
    * any other bare name is assumed to live under :data:`ORG`.
    """
    if "/" in name:
        return name
    try:
        for model_id in list_models(models_json):
            if _short(model_id) == name:
                return model_id
    except (OSError, ValueError, KeyError):
        # No usable models.json (e.g. running outside the repo) -- fall through
        # to the default-org assumption below rather than failing the lookup.
        pass
    return f"{ORG}/{name}"


def fetch_model(
    name: str,
    *,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    models_json: str = MODELS_JSON,
) -> str:
    """Download a regression model and return the path to its main ``.onnx`` file.

    ``name`` is resolved by :func:`resolve`. The download is delegated to
    ``huggingface_hub.snapshot_download`` (so it is cached); when several graphs
    are present the largest is taken as the main model, matching the regression
    workers.

    Requires ``huggingface_hub`` -- a clear ``ImportError`` is raised if it is
    missing so callers (e.g. pytest) can skip cleanly.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to fetch regression models; "
            "install it with `pip install huggingface_hub`"
        ) from e

    repo_id = resolve(name, models_json)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    onnx_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if f.endswith(".onnx")
    ]
    if not onnx_files:
        raise FileNotFoundError(f"no .onnx file found in {repo_id!r}")
    # If several graphs are present, the largest is the main model.
    return max(onnx_files, key=os.path.getsize)


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="print the curated regression model ids")
    p_list.add_argument("--short", action="store_true", help="print bare names only")

    p_resolve = sub.add_parser("resolve", help="resolve a name to a full repo id")
    p_resolve.add_argument("name")

    p_fetch = sub.add_parser("fetch", help="download a model, print its .onnx path")
    p_fetch.add_argument("name")
    p_fetch.add_argument("--local-dir", default=None)
    p_fetch.add_argument("--revision", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        for model_id in list_models():
            print(_short(model_id) if args.short else model_id)
    elif args.cmd == "resolve":
        print(resolve(args.name))
    elif args.cmd == "fetch":
        print(fetch_model(args.name, local_dir=args.local_dir, revision=args.revision))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except BrokenPipeError:
        # Downstream closed the pipe (e.g. `... | head`); exit quietly.
        sys.exit(0)
