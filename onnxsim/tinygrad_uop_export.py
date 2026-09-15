"""Exports a tinygrad ``UOp`` graph as an ONNX ``ModelProto``, one
``NodeProto`` per ``UOp``, under a custom, non-standard domain
(``"tinygrad.uop"``) -- so the result is **never runnable** by any ONNX
runtime (no runtime implements ``ALU``/``RANGE``/``BUFFER``/``REDUCE``/...
as tensor ops); this is an *inspection* format only, answering "could we
represent a UOp graph in ONNX" for
:mod:`onnxsim.webgpu_tinygrad_codegen`'s own UOp graphs (the same per-kernel
``Ops.SINK``-rooted ASTs that module's own ``_lower_tensor_program`` walks
and ``WGSLRenderer`` turns into WGSL).

``tinygrad`` is an **optional** dependency, matching the precedent set by
:mod:`onnxsim.webgpu_tinygrad_codegen` (see that module's own docstring):
nothing here runs, or is even imported, unless :func:`uop_to_onnx_model` is
actually called.

## Why bother, given it can't run

tinygrad already has its own UOp graph export/inspection story --
``VIZ=1`` records every graph-rewrite step as a ``RewriteTrace`` dataclass
and pickles it to a temp file (``tinygrad/viz/serve.py``), then serves a
bundled d3.js/dagre web UI that converts a ``UOp`` to a JSON node/edge
structure (``uop_to_json``) on the fly for rendering. Two things this
module trades for that:

- **A stable, safe container.** A pickle is a live Python object graph
  frozen to disk -- unpickling runs arbitrary code, and the format silently
  breaks across tinygrad versions whenever a pickled class's shape changes
  (``UOp``, ``ShapeTracker``, ``KernelInfo``, ...). A ``.onnx`` file is
  plain protobuf: safe to open from anywhere, and forward/backward
  tolerant of unknown fields the way protobuf always is.
- **Free generic tooling.** Netron (and any other protobuf/ONNX-aware
  viewer) renders a custom, unrecognized domain's nodes and edges
  generically -- readable graph structure with no bespoke UI to build or
  maintain, unlike tinygrad's own viz server.

The real cost: ``UOp.arg``'s type varies wildly by op (a plain ``int`` for
``DEFINE_GLOBAL``, an ``Ops`` enum member paired with an axis id for
``REDUCE``, a ``ParamArg``/``KernelInfo`` dataclass for ``PARAM``/``SINK``,
a ``ShapeTracker``/``View`` for movement ops on a pre-schedule graph, ...),
and ONNX's own ``AttributeProto`` only has native slots for
scalars/strings/tensors/graphs and lists thereof -- there is no faithful,
lossless encoding for an arbitrary Python object. This module encodes a
plain ``int``/``float``/``bool``/``str`` natively and falls back to
``repr()`` (stored as a plain string attribute) for everything else -- a
deliberately lossy escape hatch, since the goal is faithful *inspection*
(a human or a generic viewer can read every node's real dtype and a
reasonable rendering of its arg), not a lossless round trip back to a real
``UOp``. Nothing here attempts one.

## Scope

Exports exactly the ``UOp`` DAG reachable from one root (typically an
``Ops.SINK``-rooted per-kernel AST, the same one
:func:`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program` passes to
``to_program``) -- not a whole multi-kernel schedule (``schedule_linear()``'s
own result, which threads several such ASTs together via ``Ops.CALL``
nodes). Exporting one of those is just calling this once per kernel AST;
stitching multiple exported graphs into one file isn't done here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import onnx
from onnx import helper

if TYPE_CHECKING:
    from tinygrad.uop.ops import UOp

__all__ = ["DOMAIN", "uop_to_onnx_model"]

#: A deliberately unregistered, private domain string -- not
#: ``ai.onnx``/``com.microsoft``/anything a real ONNX runtime would ever
#: recognize, matching the "custom domain" convention ONNX itself documents
#: (see ``AttributeProto``/``NodeProto.domain``'s own comments in
#: ``onnx.proto``) for exactly this kind of private, non-interoperable
#: extension.
DOMAIN = "tinygrad.uop"
_OPSET_VERSION = 1


def _require_tinygrad():
    try:
        import tinygrad  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "onnxsim.tinygrad_uop_export needs the optional 'tinygrad' "
            "package: pip install tinygrad"
        ) from e


def _encode_arg(arg: object) -> Tuple[str, object] | None:
    """Picks the most natural ``AttributeProto`` encoding for a ``UOp``'s
    own ``arg`` -- see this module's own docstring for why anything beyond
    a plain scalar/string falls back to a ``repr()`` string rather than
    attempting a faithful structural encoding.
    """
    if arg is None:
        return None
    if isinstance(arg, bool):
        return ("arg_i", int(arg))
    if isinstance(arg, int):
        return ("arg_i", arg)
    if isinstance(arg, float):
        return ("arg_f", arg)
    if isinstance(arg, str):
        return ("arg_s", arg)
    return ("arg_repr", repr(arg))


def uop_to_onnx_model(root: "UOp") -> onnx.ModelProto:
    """Exports the ``UOp`` DAG reachable from ``root`` (via ``root.toposort()``,
    the same traversal :mod:`onnxsim.webgpu_tinygrad_codegen` itself uses to
    walk a scheduled kernel) as an ONNX ``ModelProto``.

    Every ``UOp`` becomes one ``NodeProto``, named/output ``u<i>`` where
    ``i`` is that ``UOp``'s position in a toposort ordering rooted at
    ``root`` (deterministic given a fixed graph, unlike Python's own
    ``id()``, which would vary run to run and make two exports of the same
    graph diff-noisy for no reason): ``op_type`` is the ``Ops`` enum
    member's own name (``"ADD"``, ``"RANGE"``, ``"SINK"``, ...), inputs are
    that ``UOp``'s own ``.src`` (by position, in order -- ``UOp.src`` is
    itself already positional/ordered), and attributes carry ``dtype`` (as
    a plain string) plus an encoded ``arg`` (see :func:`_encode_arg`) when
    the ``UOp`` has one.

    :param root: typically an ``Ops.SINK``-rooted per-kernel AST (a single
            ``Ops.CALL``'s own ``src[0]`` from a scheduled program -- see
            this module's own docstring on scope).
    :returns: a standalone ``ModelProto`` -- one graph, one custom-domain
            opset import, no standard-domain nodes at all. Never
            executable by any ONNX runtime; see this module's own
            docstring for why that's fine for what this is for.
    """
    _require_tinygrad()

    order = list(root.toposort())
    index: Dict["UOp", int] = {u: i for i, u in enumerate(order)}

    def name_for(u: "UOp") -> str:
        return f"u{index[u]}"

    nodes = []
    for u in order:
        kwargs: Dict[str, object] = {"dtype": str(u.dtype)}
        encoded = _encode_arg(u.arg)
        if encoded is not None:
            kwargs[encoded[0]] = encoded[1]
        nodes.append(
            helper.make_node(
                op_type=u.op.name,
                inputs=[name_for(s) for s in u.src],
                outputs=[name_for(u)],
                name=name_for(u),
                domain=DOMAIN,
                **kwargs,  # type: ignore[arg-type]  # make_node's **kwargs really is Any at runtime
            )
        )

    # ONNX's checker requires every graph output to carry a fully-formed
    # `type` (elem_type *and* a shape, even an empty/scalar one) -- there is
    # no "unknown/opaque" placeholder it accepts, so this uses a scalar
    # FLOAT as a formality; it is not a claim about the root UOp's real
    # dtype (already recorded faithfully as that node's own "dtype"
    # attribute above) or shape.
    graph = helper.make_graph(
        nodes,
        "tinygrad_uop_graph",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info(name_for(root), onnx.TensorProto.FLOAT, [])
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 1),
            helper.make_opsetid(DOMAIN, _OPSET_VERSION),
        ],
        producer_name="onnxsim.tinygrad_uop_export",
    )
    model.ir_version = onnx.IR_VERSION
    return model
