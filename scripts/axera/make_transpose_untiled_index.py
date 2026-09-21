"""Build the untiled-Transpose template index from a sweep root.

For every complete ``(R, q)`` block (all seven ``C = 8q-7 .. 8q-1`` built, equal
MCode length) this fits the field map on five shapes (``C mod 8`` in 1, 3, 4, 5
and 7), holds out the other two (``C mod 8`` in 2 and 6), scores the prediction
byte for byte against the held-out builds, and writes:

* ``fixtures/transpose_untiled/index.json``: per block, the template fixture
  name, the fitted field map, and the held-out score;
* one gzipped compiled template per block (the ``C mod 8 == 4`` build);
* ``fixtures/transpose_untiled/oracles.json.gz``: the held-out builds' MCode,
  which the tests compare emitted models against.

Usage::

    make_transpose_untiled_index.py WORK_ROOT [WORK_ROOT ...]
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import transpose_fields as tf  # noqa: E402

_OUT = os.path.join(_HERE, "fixtures", "transpose_untiled")
_NAME = re.compile(r"^A_1x1x(\d+)x(\d+)$")
_TRAIN = (1, 3, 4, 5, 7)
_HELD = (2, 6)


def _path(roots: list[str], r: int, c: int) -> str | None:
    for root in roots:
        p = os.path.join(root, f"A_1x1x{r}x{c}", "out", "compiled.axmodel")
        if os.path.exists(p):
            return p
    return None


def _neu(model: onnx.ModelProto) -> bytes:
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


def main(roots: list[str]) -> int:
    blocks = set()
    for root in roots:
        for name in os.listdir(root):
            match = _NAME.match(name)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                if c % 8:
                    blocks.add((r, (c + 7) // 8))
    os.makedirs(_OUT, exist_ok=True)
    index, oracles = [], {}
    skipped = []
    for r, q in sorted(blocks):
        cs = {res: 8 * (q - 1) + res for res in range(1, 8)}
        paths = {res: _path(roots, r, c) for res, c in cs.items()}
        if any(p is None for p in paths.values()):
            skipped.append((r, q, "incomplete"))
            continue
        models = {
            res: onnx.load(p, load_external_data=False) for res, p in paths.items()
        }
        code = {res: _neu(m) for res, m in models.items()}
        if len({len(b) for b in code.values()}) != 1:
            skipped.append((r, q, "unequal lengths"))
            continue
        train = {(r, cs[res]): code[res] for res in _TRAIN}
        try:
            fields = tf.fit(train, tie_break=True)
        except ValueError as exc:
            skipped.append((r, q, f"fit: {exc}"))
            continue
        template = code[4]
        held = {(r, cs[res]): code[res] for res in _HELD}
        exact, bad = tf.score(template, fields, held)
        name = f"transpose_R{r}_q{q}.axmodel.gz"
        with gzip.open(os.path.join(_OUT, name), "wb", compresslevel=9) as f:
            f.write(models[4].SerializeToString())
        index.append(
            {
                "R": r,
                "q": q,
                "template": name,
                "template_C": cs[4],
                "mcode_len": len(template),
                "fields": {str(p): list(v) for p, v in sorted(fields.items())},
                "held_out": {"shapes": [list(s) for s in sorted(held)], "exact": exact},
            }
        )
        for shape, blob in held.items():
            oracles[f"{shape[0]},{shape[1]}"] = base64.b64encode(blob).decode()
        print(
            f"R={r:2d} q={q:2d}: {len(fields)} fields, held-out exact {exact}/{len(held)}"
        )
    with open(os.path.join(_OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)
    with gzip.open(os.path.join(_OUT, "oracles.json.gz"), "wt", compresslevel=9) as f:
        json.dump(oracles, f, sort_keys=True)
    for r, q, why in skipped:
        if why != "incomplete":
            print(f"skipped R={r} q={q}: {why}")
    print(f"{len(index)} blocks indexed, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
