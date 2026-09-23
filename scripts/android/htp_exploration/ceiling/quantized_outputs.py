#!/usr/bin/env python3
"""Make the backbone emit its 14 outputs as the uint8 tensors they already are internally.

Every backbone output is `DequantizeLinear(QuantizeLinear(x))`: the graph quantizes, then
dequantizes to fp32 only to hand the result out. On the HTP that final dequantize plus writing
~75 MB of fp32 (the P2 FPN map alone is 256x200x272x4 = 55.7 MB) is the single largest remaining
cost. This rewrite points each graph output at the uint8 Q tensor instead and records its
(scale, zero_point) in outputs_qparams.json so the consumer (ORT on the CPU, or this project's
DSP kernels, which already read uint8 directly) can dequantize itself. Numerically lossless: the
fp32 values were exactly scale*(q-zp) anyway.

usage: quantized_outputs.py <in.onnx> <out.onnx>
"""
import json
import sys
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def main():
    m = onnx.load(sys.argv[1])
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    qparams, new_outs = {}, []
    for o in g.output:
        dq = prod[o.name]
        assert dq.op_type == "DequantizeLinear", o.name
        q = dq.input[0]
        zp = init[dq.input[2]]
        qparams[q] = {"float_output": o.name, "scale": float(init[dq.input[1]]), "zero_point": int(zp)}
        shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        new_outs.append(helper.make_tensor_value_info(q, helper.np_dtype_to_tensor_dtype(zp.dtype), shape))
    del g.output[:]
    g.output.extend(new_outs)
    # drop DQs whose only purpose was producing a graph output
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    for n in [n for n in g.node if n.op_type == "DequantizeLinear" and n.output[0] not in used]:
        g.node.remove(n)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    Path(sys.argv[2]).with_name("outputs_qparams.json").write_text(json.dumps(qparams, indent=1))
    print(f"{len(new_outs)} outputs now uint8")


if __name__ == "__main__":
    main()
