#!/usr/bin/env python3
"""Emit NPU mcode for tinygrad-traced graphs.

Pipeline: a tinygrad ``Tensor`` graph is traced to its UOp pattern,
matched by one of the ``trace_*`` functions below, and (for Neg so far)
emitted as an mcode stream assembled from a reference Pulsar2 build with
caller-chosen output scales (found by value as float32 words -- Neg
carries its output scale as four stride-7 copies).

``trace_neg``/``trace_add``/``trace_mul`` match single- or two-node UOp
patterns (unary negation lowers to ``MUL(x, CONST(-1.0))``; a genuine
two-tensor add or multiply to ``ADD(a, b)``/``MUL(a, b)`` with neither
operand a ``CONST`` -- the two ``Ops.MUL`` matchers are mutually
exclusive by that const check). ``trace_relu``/``trace_sigmoid`` match
deeper compound patterns tinygrad lowers these ops to, since neither has
a dedicated UOp: Relu is ``WHERE(CMPLT(CONST(0.0), x), x, CONST(0.0))``
(a compare-and-select, with an operand-*identity* check -- not just
shape/dtype equality -- that the ``x`` compared against 0 is the same
UOp used as the true branch), and Sigmoid is
``RECIPROCAL(ADD(CONST(1.0), EXP2(MUL(x, CONST(-log2(e))))))`` (the
standard ``e^-x = 2^(-x*log2(e))`` rewrite, checked by exact float
equality against ``-math.log2(math.e)``). Every pattern here was
confirmed by direct UOp inspection against a real tinygrad install, not
assumed from tinygrad's Python source -- these lowerings can and do
change across tinygrad versions.

None of the four new matchers has an ``emit_*`` counterpart yet (only
Neg does, via ``emit_neg``) -- see each function's own docstring and
tests/test_axera_tiny_emit.py for what tracing alone does and doesn't
give you.

``patch_mul_scales`` extends the same reference-patch idea to two-input
Mul streams' input side (sites A/C/B -- see
tests/test_axera_mcode_reciprocal.py for the field map): given the
reference build's recorded scales and the target scales, it rewrites the
input reciprocal and requant slots by value, verifying each family forms
its exact stride run. ``patch_mul_output_quad`` does the same for the
output-side scale quad (``03 <f32(z_scale)> 81 <tag2>`` x4 stride 7--
``tag2`` varies build to build, unlike the always-``81`` byte before it;
not patched, not relied on). ``patch_mul_zp_x`` patches x's zero point,
but *only* when the reference build happens to use the one zero-point
form this project has actually decoded (see below) -- it raises rather
than silently doing nothing when it doesn't apply, since whether it
applies is not predictable in advance. ``patch_conv_zp_x`` is the same
patch for Conv (the literal unit was confirmed identical across ops);
see its own docstring for a caveat ``patch_mul_zp_x`` does not carry --
patching this unit alone was checked against a real second Conv build
with x_scale held bit-identical and did *not* reproduce that build's
mcode elsewhere in the stream (~20 bytes beyond the unit and the
shape's own noise floor also move with zp_x), unlike the scale-family
patches below, which a real hardware test confirmed reproduce a
rebuild bit-exactly.

What none of this touches: the S-unit programs themselves (magnitude-
adaptive shape, unmodeled ISA), the manifest string table (tensor-name
order varies build to build), y's zero point (no literal encoding found
for it at all, decoded or not), z's own zero point (see the hardware
result below -- it did not need patching, at least once), or x's zero
point when the reference build doesn't use the literal form -- which is
the common case, not an edge case (confirmed non-predictable from zp_x's
value at any magnitude, see ``TestZpXLiteralByteWhenPresent`` in the
test file). Because of that last point, and because z_scale changing
generally moves far more of the stream than the five families patched
here (A, B, C, the output quad, zp_x -- z's own calibration range shifts
the S-unit programs' internal
constants throughout, not just the named slots -- confirmed by diffing
same-shape builds at different z scales: over a thousand bytes move),
**full-stream equality after patching is not a goal and should not be
expected**; what is verified is that each named family lands on the
target build's own bytes for that family.

Status (2026-09-16): scale, output-quad and (when present) zp_x words
are mapped, verified offline (round-trip exactly, pass ``mcode.check``),
and confirmed twice on real AX650N hardware
(tests/test_axera_mul_emit_hardware.py): patching a reference build's
scales + output quad + zp_x (17 bytes total, in that one build pair)
reproduced a real rebuild's device output *bit-exactly*, not just within
quantization noise, and neither zp_y nor zp_z needed touching for that
result to hold. This supersedes an earlier note here about the emitted
stream running ~0.19-vs-0.006 against ORT -- that check pre-dated the
output-quad and zp_x work and used a different code path (``emit_neg``,
not the Mul patch functions); it has not been repeated against these.

**Practical recommendation, also hardware-confirmed**: if the calibration
data can be chosen (as opposed to given), keep it non-negative for both
inputs -- MinMax clipping then guarantees zp_x = zp_y = 0 (see
``TestSiteBFormSelectorIsZpX`` in the test file), which sidesteps
``patch_mul_zp_x`` and its "does this build even use the literal form"
uncertainty entirely: only ``patch_mul_scales`` + ``patch_mul_output_quad``
are needed, and that combination also reproduces a real rebuild
bit-exactly (second test in the hardware file). Two data points, not a
general proof -- see the hardware test file's own docstring for exactly
what is and isn't covered. tinygrad itself is an optional, lazily-imported
dependency: everything else here needs only
``numpy``.
"""

from __future__ import annotations

import struct


def trace_neg(tensor):
    """Match a tinygrad tensor holding unary negation.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``MUL(x, const(-1.0))`` (what ``-t`` lowers to), else raises
    ``ValueError``. tinygrad is imported lazily so this module stays
    importable without it.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.MUL or len(uop.src) != 2:
        raise ValueError(f"not a multiply: {uop.op}")
    data, const = uop.src
    if const.op is not Ops.CONST or float(const.arg) != -1.0:
        raise ValueError(f"not a multiply-by-minus-one: {const}")
    shape = list(data.shape)
    return {"shape": shape, "dtype": str(data.dtype)}


def trace_add(tensor):
    """Match a tinygrad tensor holding elementwise two-tensor addition.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``ADD(a, b)`` with neither operand a ``CONST`` (a genuine
    two-tensor add, not a scalar-add lowering, which would need its own
    matcher the way ``trace_neg`` needs one distinct from a general
    two-tensor multiply) -- else raises ``ValueError``.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.ADD or len(uop.src) != 2:
        raise ValueError(f"not an add: {uop.op}")
    a, b = uop.src
    if a.op is Ops.CONST or b.op is Ops.CONST:
        raise ValueError("a scalar add, not a two-tensor add")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def trace_mul(tensor):
    """Match a tinygrad tensor holding elementwise two-tensor multiply.

    Returns ``{"shape": [...], "dtype": ...}`` if the tensor's UOp graph
    is ``MUL(a, b)`` with neither operand a ``CONST``. This is the
    complement of ``trace_neg``'s pattern (``MUL(x, CONST(-1.0))``) over
    the same ``Ops.MUL`` space -- a scalar multiply-by-constant (Neg's
    shape, or any other scalar multiply) is rejected here and matched by
    ``trace_neg`` instead when the constant happens to be -1.0; any other
    scalar multiply matches neither and needs its own matcher, not added
    here since this project has no confirmed byte-level scalar-multiply
    encoding yet.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.MUL or len(uop.src) != 2:
        raise ValueError(f"not a multiply: {uop.op}")
    a, b = uop.src
    if a.op is Ops.CONST or b.op is Ops.CONST:
        raise ValueError("a scalar multiply, not a two-tensor multiply")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def trace_relu(tensor):
    """Match a tinygrad tensor holding ReLU.

    tinygrad has no dedicated Relu UOp -- ``.relu()`` lowers to a
    compare-and-select, ``WHERE(CMPLT(CONST(0.0), x), x, CONST(0.0))``,
    confirmed by direct UOp inspection (not assumed from tinygrad's
    Python source, which can and does change the lowering across
    versions). Matching it means checking that three-node shape *and*
    that the ``x`` referenced in the comparison is the identical UOp
    object used as the true-branch -- tinygrad's UOp graph shares
    subexpression nodes, so ``is`` identity is the correct check, not
    structural equality (two different tensors could legitimately have
    identical shape/dtype without being the same operand).
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc

    uop = tensor.uop
    if uop.op is not Ops.WHERE or len(uop.src) != 3:
        raise ValueError(f"not a where: {uop.op}")
    cmp, true_branch, false_branch = uop.src
    if cmp.op is not Ops.CMPLT or len(cmp.src) != 2:
        raise ValueError(f"not a relu (where's condition isn't a cmplt): {cmp.op}")
    zero_lhs, x = cmp.src
    if zero_lhs.op is not Ops.CONST or float(zero_lhs.arg) != 0.0:
        raise ValueError("not a relu (cmplt's left side isn't the constant 0.0)")
    if true_branch is not x:
        raise ValueError("not a relu (where's true branch isn't cmplt's operand)")
    if false_branch.op is not Ops.CONST or float(false_branch.arg) != 0.0:
        raise ValueError("not a relu (where's false branch isn't the constant 0.0)")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def trace_sigmoid(tensor):
    """Match a tinygrad tensor holding Sigmoid.

    Like Relu, tinygrad has no dedicated Sigmoid UOp: ``.sigmoid()``
    lowers to ``RECIPROCAL(ADD(CONST(1.0), EXP2(MUL(x, CONST(c)))))``
    where ``c`` is exactly ``-log2(e)`` (the standard
    ``e^-x = 2^(-x*log2(e))`` rewrite so the hardware/software exp
    lowers through ``EXP2`` rather than a natural-base ``Ops.EXP``),
    confirmed by direct UOp inspection and an exact float equality check
    against ``-math.log2(math.e)`` -- not a tolerance-based comparison,
    since tinygrad computes this constant once at trace time and every
    build should reproduce the identical float64-rounded-to-float32 bit
    pattern. The deepest pattern of the four traced here (four nested
    ops); unlike Relu's compare-and-select, no operand identity check is
    needed beyond following the chain down to a single ``x``.
    """
    try:
        from tinygrad.uop.ops import Ops
    except ImportError as exc:
        raise ImportError(
            "tracing needs the tinygrad package (pip install tinygrad)"
        ) from exc
    import math

    uop = tensor.uop
    if uop.op is not Ops.RECIPROCAL or len(uop.src) != 1:
        raise ValueError(f"not a reciprocal: {uop.op}")
    (add_node,) = uop.src
    if add_node.op is not Ops.ADD or len(add_node.src) != 2:
        raise ValueError(
            f"not a sigmoid (reciprocal's operand isn't an add): {add_node.op}"
        )
    one_const, exp2_node = add_node.src
    if one_const.op is not Ops.CONST or float(one_const.arg) != 1.0:
        raise ValueError("not a sigmoid (add's first operand isn't the constant 1.0)")
    if exp2_node.op is not Ops.EXP2 or len(exp2_node.src) != 1:
        raise ValueError(
            f"not a sigmoid (add's second operand isn't an exp2): {exp2_node.op}"
        )
    (mul_node,) = exp2_node.src
    if mul_node.op is not Ops.MUL or len(mul_node.src) != 2:
        raise ValueError(
            f"not a sigmoid (exp2's operand isn't a multiply): {mul_node.op}"
        )
    x, c = mul_node.src
    if c.op is not Ops.CONST or float(c.arg) != -math.log2(math.e):
        raise ValueError("not a sigmoid (multiply's constant isn't -log2(e))")
    return {"shape": list(uop.shape), "dtype": str(tensor.dtype)}


def find_scale_words(mcode: bytes, scale: float) -> list:
    """Offsets of every float32 occurrence of ``scale`` in the stream.

    Neg carries its output scale as four stride-7 copies; this returns
    wherever the value literally occurs so the caller can decide which
    copies are output-side. No layout assumptions beyond the byte match.
    """
    if isinstance(mcode, bytearray):
        mcode = bytes(mcode)
    pat = struct.pack("<f", float(scale))
    return [i for i in range(len(mcode) - 3) if mcode[i : i + 4] == pat]


def minmax_scale(samples) -> float:
    """Pulsar2's output-scale formula, verified to ~1e-10 against six
    real Neg builds: ``(max - min) / 255`` over the calibration samples
    (computed in float64). The zero point formula is still open (floor
    fits 4/9 builds), so this returns the scale only."""
    import numpy as np

    flat = np.concatenate([np.asarray(s).reshape(-1) for s in samples]).astype(
        np.float64
    )
    return float((flat.max() - flat.min()) / 255.0)


def emit_neg(reference_mcode: bytes, old_scale: float, new_scale: float) -> bytes:
    """Replace every float32 occurrence of ``old_scale`` with ``new_scale``.

    Both ends must be exactly representable (pass ``float(np.float32(x))``
    -- a float64 with extra digits never matches). Returns bytes that
    decode and round-trip exactly like the reference; whether they
    *compute* identically is a device question (see module docstring).
    """
    import struct as _struct

    new = _struct.pack("<f", float(new_scale))
    offsets = find_scale_words(reference_mcode, float(old_scale))
    if not offsets:
        raise ValueError(f"scale {old_scale!r} occurs nowhere: wrong reference?")
    out = bytearray(reference_mcode)
    for off in offsets:
        out[off : off + 4] = new
    return bytes(out)


def _strided_run(mcode: bytes, pattern: bytes, stride: int, count: int = 4) -> list:
    """Offsets where ``pattern`` occurs as exactly ``count`` stride-run copies.

    Raises ``ValueError`` unless the occurrences are exactly ``count`` and
    land on a perfect stride grid -- an incidental byte collision anywhere
    else in the stream fails loudly instead of patching half a slot family.
    """
    hits = [
        i
        for i in range(len(mcode) - len(pattern) + 1)
        if mcode[i : i + len(pattern)] == pattern
    ]
    if len(hits) != count or any(b - a != stride for a, b in zip(hits, hits[1:])):
        raise ValueError(
            f"pattern {pattern.hex()} hits {hits}: not a stride-{stride} x{count} run"
        )
    return hits


def patch_mul_scales(reference_mcode: bytes, old_scales, new_scales) -> bytes:
    """Rewrite a two-input Mul stream's input-side scale slots by value.

    ``old_scales``/``new_scales`` are ``(x_scale, y_scale, z_scale)``
    triples -- the reference build's recorded MinMax scales and the
    target's. Three slot families move (see
    tests/test_axera_mcode_reciprocal.py):

    - site A (stride-8 x4): float32(1/x_scale), the x quant multiplier;
    - site C (stride-8 x4): float32(z_scale/(x_scale*y_scale)), the
      integer requant multiplier;
    - site B: float32(1/y_scale) x4 at stride 7 when the reference uses
      the full S-unit form, else the short form's low 3 bytes x4 at
      stride 6 (tag byte preserved). The reference's form is kept: this
      patches values, it does not recompile programs.

    Every family is stride-verified; a missing or ambiguous family
    raises. Output quads, S-unit programs, the string table and zero
    points are untouched (see module docstring).
    """
    old_x, old_y, old_z = (float(s) for s in old_scales)
    new_x, new_y, new_z = (float(s) for s in new_scales)
    # Locate every family on the pristine reference first: patching one
    # family must never disturb another family's search (overlapping
    # values across families would otherwise corrupt the later lookup).
    edits = []

    def locate(old_value: float, new_value: float, stride: int, width: int = 4):
        old_pat = struct.pack("<f", old_value)[:width]
        new_pat = struct.pack("<f", new_value)[:width]
        for off in _strided_run(reference_mcode, old_pat, stride):
            edits.append((off, new_pat))

    locate(1.0 / old_x, 1.0 / new_x, 8)
    locate(old_z / (old_x * old_y), new_z / (new_x * new_y), 8)
    try:
        locate(1.0 / old_y, 1.0 / new_y, 7)
    except ValueError:
        locate(1.0 / old_y, 1.0 / new_y, 6, width=3)
    out = bytearray(reference_mcode)
    for off, new_pat in edits:
        out[off : off + len(new_pat)] = new_pat
    return bytes(out)


def patch_mul_output_quad(
    reference_mcode: bytes, old_z_scale: float, new_z_scale: float
) -> bytes:
    """Rewrite a Mul stream's output scale quads by value.

    The output-side counterpart to ``patch_mul_scales``'s input-side
    families: four copies of ``03 <f32(z_scale)> 81 <tag2>`` at stride 7
    (see ``TestOutputScaleQuads`` in tests/test_axera_mcode_reciprocal.py).
    Verifies the ``03``/``81`` framing on every copy before patching, on
    top of ``_strided_run``'s count/stride check, since those two bytes
    are cheap to confirm and a false match here would silently leave the
    output at the old scale. The second tag byte is NOT checked: it read
    ``82`` on every fixture ``TestOutputScaleQuads`` was written against,
    which that test's docstring stated as if constant, but two more real
    builds (zp_x=33/35, output zp 30/32) came back ``80`` instead -- only
    the *value* (this function's job) and the ``03``/``81`` framing hold
    across all six builds gathered so far. This function preserves
    whatever the second tag byte already is; it never depends on its
    value.
    """
    old_pat = struct.pack("<f", float(old_z_scale))
    new_pat = struct.pack("<f", float(new_z_scale))
    hits = _strided_run(reference_mcode, old_pat, 7)
    for off in hits:
        lead, tag1 = reference_mcode[off - 1], reference_mcode[off + 4]
        if lead != 0x03 or tag1 != 0x81:
            raise ValueError(
                f"quad @{off}: frame {lead:02x}/{tag1:02x} is not 03../81."
            )
    out = bytearray(reference_mcode)
    for off in hits:
        out[off : off + 4] = new_pat
    return bytes(out)


def patch_output_quad(
    reference_mcode: bytes, old_z_scale: float, new_z_scale: float
) -> bytes:
    """Rewrite the output-scale quad for Gemm/Conv/MatMul (and Mul).

    ``patch_mul_output_quad`` only recognizes Mul's own bare frame (``03
    <f32(z_scale)> 81 <tag2>`` x4 stride 7). This session's later decode
    work found the *identical* mechanism on Gemm, Conv and MatMul too
    (``tests/test_axera_output_scale_quad_generalizes.py``,
    ``tests/test_axera_gemm_output_quad.py``), framed with a `05 50 0f`
    lead-in before the first copy instead: ``05 50 0f <f32(z_scale)> 81
    <tag2> 03`` x4 stride 7. This function handles both frames by
    checking for the lead-in first and falling back to Mul's bare-``03``
    frame when it's absent, so one function covers every op this project
    has found the quad on.

    Same defensive posture as ``patch_mul_output_quad``: verifies the
    stride-7 x4 run and the ``0x81`` tail byte (and the lead-in, when
    present) before patching, but never depends on ``tag2`` -- confirmed
    build-specific across every op checked, not a shared constant.
    Preserves whatever ``tag2`` and the trailing 0x03/0x83 high-bit byte
    already are; this only ever rewrites the 4-byte float.

    Verified (2026-09-17), compile-only, no device access needed:
    patching `gemm_1x8x8_tb0.mcode.gz`'s quad to `gemm_1x8x8_tb1.mcode.gz`'s
    real output_scale reproduces `tb1`'s own mcode byte-for-byte outside
    this project's already-known ``~295-330`` noise zone -- see
    ``tests/test_axera_output_quad_generator.py``.
    """
    old_pat = struct.pack("<f", float(old_z_scale))
    new_pat = struct.pack("<f", float(new_z_scale))
    hits = _strided_run(reference_mcode, old_pat, 7)
    lead_in = reference_mcode[hits[0] - 3 : hits[0]]
    has_lead_in = lead_in == bytes.fromhex("05500f")
    if not has_lead_in and reference_mcode[hits[0] - 1] != 0x03:
        raise ValueError(
            f"quad @{hits[0]}: lead-in {lead_in.hex()} is neither the"
            " generalized 05500f frame nor Mul's bare 03 frame"
        )
    for off in hits:
        if reference_mcode[off + 4] != 0x81:
            raise ValueError(
                f"quad @{off}: tail byte {reference_mcode[off + 4]:02x} is not 81"
            )
    out = bytearray(reference_mcode)
    for off in hits:
        out[off : off + 4] = new_pat
    return bytes(out)


def patch_mul_zp_x(reference_mcode: bytes, old_zp_x: int, new_zp_x: int) -> bytes:
    """Rewrite x's zero point, where the literal-byte form is present.

    The unit ``02 10 1b <zp_x> 83 36`` carries zp_x verbatim as its
    fourth byte in some Mul builds (see ``TestZpXLiteralByteWhenPresent``)
    -- but whether a given build uses this form is not predictable from
    zp_x's value at any magnitude; roughly as many builds use one of two
    other, still-undecoded forms instead (``TestZpXImmediateRegion``).
    This function only ever does the one thing it has evidence for:
    raises ``ValueError`` if the reference build does not carry
    ``old_zp_x`` in this exact form, rather than silently leaving zp_x
    unpatched or guessing at the opaque forms' encoding.
    """
    if not 0 <= old_zp_x <= 255 or not 0 <= new_zp_x <= 255:
        raise ValueError(f"zp_x must be a uint8: old={old_zp_x!r} new={new_zp_x!r}")
    old_unit = bytes.fromhex("02101b") + bytes([old_zp_x]) + bytes.fromhex("8336")
    hits = [
        i
        for i in range(len(reference_mcode) - len(old_unit) + 1)
        if reference_mcode[i : i + len(old_unit)] == old_unit
    ]
    if len(hits) != 1:
        raise ValueError(
            f"literal zp_x unit for {old_zp_x} not found exactly once"
            f" (found {len(hits)}) -- this reference build likely uses one"
            " of the opaque, undecoded forms instead"
        )
    out = bytearray(reference_mcode)
    out[hits[0] + 3] = new_zp_x
    return bytes(out)


def patch_conv_zp_x(reference_mcode: bytes, old_zp_x: int, new_zp_x: int) -> bytes:
    """Rewrite Conv's input zero point, where the literal-byte form is
    present -- the same ``02 10 1b <zp_x> 83 36`` unit ``patch_mul_zp_x``
    patches for Mul, confirmed to generalize byte-for-byte to Conv at the
    identical framing (``tests/test_axera_zpx_generalizes.py``). Delegates
    to ``patch_mul_zp_x`` directly: the unit and its patch are op-agnostic
    once the literal form is present, so there is nothing Conv-specific to
    do here beyond documenting Conv's own generation-scope caveat below.

    **This patches the VALUE inside an already-literal-form unit. It does
    NOT, and per current knowledge cannot, predict whether a genuinely
    fresh Conv build at ``new_zp_x`` would itself land in this literal
    form or one of the two still-undecoded opaque forms** -- presence is
    non-monotone in zp_x for both Mul and Conv (see
    ``TestZpXLiteralByteWhenPresent``), so this function's success at
    finding ``old_zp_x``'s unit says nothing about whether ``new_zp_x``
    "should" have used the same form; it only says the reference happens
    to hold both bytes in the one slot this project knows how to write.

    **A second, newly-confirmed caveat, found verifying this function
    against a real second Conv build (2026-09-17):** even holding
    x_scale exactly fixed (bit-identical between the two builds, not
    just close), patching *only* this 6-byte unit does not reproduce a
    real second build's mcode elsewhere in the stream -- roughly twenty
    bytes beyond this unit and beyond the shape's own ~3-byte rebuild
    noise floor also move with zp_x alone, at offsets this project has
    not decoded. Callers should not expect (and this function does not
    claim) that patching this unit alone reproduces a real rebuild
    bit-exactly, the way Mul's *scale*-family patches were confirmed to
    on real hardware (``tests/test_axera_mul_emit_hardware.py``) --
    zp_x's own patch has not had, and does not currently pass, that same
    bar. See ``tests/test_axera_conv_zpx_generator.py`` for the exact
    build pair and byte counts this was checked against.
    """
    return patch_mul_zp_x(reference_mcode, old_zp_x, new_zp_x)
