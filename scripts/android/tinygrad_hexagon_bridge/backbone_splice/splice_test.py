"""Splice the chunked kernel into a real qnn.conv2d build via te.extern, replacing TVM's own
compute for the matching (cin=64,cout=256,1x1,stride=1) shape. Correctness check against the
same conv computed normally by TVM's own (unmodified) schedule, small spatial size for speed."""
import os
import numpy as np
import tvm
import tvm.contrib.hexagon
from tvm import te, relay
import tvm.topi.hexagon.conv2d as hexconv
import tvm.topi.hexagon as hexconv_pkg

TARGET_SHAPE = (64, 256)  # (cin, cout) to intercept

_orig_compute = hexconv.conv2d_NCHWc_int8

def chunked_compute(data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="int32"):
    ic_chunks = int(data.shape[1])
    ic_bn = int(data.shape[4])
    oc_chunks = int(kernel.shape[0])
    cin, cout = ic_chunks * ic_bn, oc_chunks * 32
    if (cin, cout) != TARGET_SHAPE or int(kernel.shape[2]) != 1 or int(kernel.shape[3]) != 1 or tuple(int(s) for s in stride) != (1, 1):
        return _orig_compute(data, kernel, stride, padding, dilation, layout, out_layout, out_dtype)

    H, W = int(data.shape[2]), int(data.shape[3])
    out_shape = (1, oc_chunks, H, W, 32)

    def fcompute(ins, outs):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "void", "hex_gemm_chunked_64_256", outs[0].data, ins[0].data, ins[1].data))

    return te.extern(out_shape, [data, kernel], fcompute, name="conv2d_chunked_64_256", dtype=out_dtype)

hexconv.conv2d_NCHWc_int8 = chunked_compute
hexconv_pkg.conv2d_NCHWc_int8 = chunked_compute

# also need the schedule to just leave the extern call alone (default_schedule via
# te.create_schedule should already work fine for an extern op -- no vrmpy tensorize needed
# since our own kernel handles all of that internally)

cin, cout = TARGET_SHAPE
H, W = 20, 20  # small for fast correctness check
ishape = [1, cin, H, W]
wshape = [cout, cin, 1, 1]
rng = np.random.default_rng(3)
weight = rng.integers(-40, 40, wshape).astype("int8")
data = relay.var("data", shape=ishape, dtype="uint8")
conv = relay.qnn.op.conv2d(
    data, relay.const(weight), relay.const(114, "int32"), relay.const(0, "int32"),
    relay.const(0.02, "float32"), relay.const(0.005, "float32"), kernel_size=(1, 1),
    channels=cout, strides=(1, 1), padding=(0, 0), out_dtype="int32",
)
mod = tvm.IRModule.from_expr(relay.Function([data], conv))
image = rng.integers(0, 100, ishape).astype("uint8")

target = tvm.target.Target(tvm.target.hexagon("v73"), host=tvm.target.hexagon("v73"))
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target)
print("BUILD OK (spliced)")
def fcompile_wrapper(so_name, objs, **kwargs):
    # link_shared is TVM-FFI-wrapped (PackedFunc), which doesn't accept **kwargs directly --
    # export_library() forwards its own kwargs (extra_args=...) as **kwargs to fcompile, so a
    # plain-Python wrapper is needed to convert back to link_shared's positional 3rd arg.
    # export_library() also may hand us a devc.c (its module-packing fallback for imported
    # submodules) directly, expecting fcompile to compile it -- link_shared is a pure linker
    # (hexagon-link), so compile any .c inputs to .o with the Hexagon toolchain ourselves first.
    import subprocess

    hexagon_clang = os.path.join(os.environ["HEXAGON_TOOLCHAIN"], "bin", "hexagon-clang")
    compiled_objs = []
    for obj in objs:
        if obj.endswith(".c"):
            out_o = obj[:-2] + ".o"
            subprocess.run(
                [hexagon_clang, "-c", "-O2", "-fPIC", "-mcpu=hexagonv73", "-mhvx=v73",
                 "-mhvx-length=128b", "-o", out_o, obj],
                check=True,
            )
            compiled_objs.append(out_o)
        else:
            compiled_objs.append(obj)
    return tvm.contrib.hexagon.tools.link_shared(so_name, compiled_objs, kwargs.get("extra_args"))

lib.export_library(
    "spliced_test.so",
    fcompile=fcompile_wrapper,
    addons=["kernel_chunked_64_256.o"],
    extra_args={"hex_arch": "v73"},
)
print("LINKED OK")

# --- verify correctness on real hardware, against the same conv computed by stock TVM ---
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker

hexconv.conv2d_NCHWc_int8 = _orig_compute
hexconv_pkg.conv2d_NCHWc_int8 = _orig_compute
with tvm.transform.PassContext(opt_level=3):
    lib_stock = relay.build(mod, target=target)

tracker = Tracker(host="127.0.0.1", port=9187)
launcher = HexagonLauncher("239dbd8f", rpc_info={"rpc_tracker_host": "127.0.0.1", "rpc_tracker_port": 9187,
    "rpc_server_port": 7067, "workspace_base": "/data/local/tmp/tg_splice_verify", "adb_server_socket": None})
try:
    launcher.start_server()
    # Separate sessions for spliced vs. stock -- mixing a manually-loaded module with
    # get_executor_from_factory() in one session causes spurious hexagon_rpc_send failures
    # unrelated to either module (found independently in this project's elementwise-add work).
    with launcher.create_session() as session:
        remote_path = session.upload("spliced_test.so", "spliced_test.so")
        graph_mod = session.load_module(str(remote_path))
        print("load_module OK")
        gm = tvm.contrib.graph_executor.create(lib.get_graph_json(), graph_mod, session.device)
        print("graph_executor.create OK")
        gm.load_params(tvm.runtime.save_param_dict(lib.get_params()))
        print("load_params OK")
        gm.set_input("data", image)
        print("set_input OK")
        gm.run()
        print("gm.run() OK (spliced)")
        out_spliced = gm.get_output(0).numpy().copy()

    with launcher.create_session() as session:
        gm2 = session.get_executor_from_factory(lib_stock)
        gm2.load_params(tvm.runtime.save_param_dict(lib_stock.get_params()))
        gm2.set_input("data", image)
        gm2.run()
        out_stock = gm2.get_output(0).numpy().copy()

    match = bool(np.array_equal(out_spliced, out_stock))
    print("spliced vs stock TVM, real hardware, bit-exact match:", match)
    if not match:
        diff = out_spliced.astype(np.int64) - out_stock.astype(np.int64)
        print("max abs diff", np.abs(diff).max(), "mismatched", np.count_nonzero(diff), "/", diff.size)
finally:
    launcher.stop_server()
    tracker.terminate()
