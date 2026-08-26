#!/usr/bin/env bash
# Build and run bench/graph_shape_inference_teardown_bench.cpp -- the
# microbenchmark for the arena change in ProcessNode
# (third_party/onnx/onnx/common/graph_shape_inference.cc).
#
# It needs the onnx C++ proto headers/lib and protobuf. Both are already built
# when you build onnxsim, so the easiest path is to point this at that build:
#
#   ONNX_INCLUDE  dirs with onnx/onnx_pb.h and the generated onnx/onnx.pb.h
#                 (colon-separated; e.g. third_party/onnx and the build tree)
#   ONNX_LIB      dir(s) containing libonnx / libonnx_proto (colon-separated)
#   PROTOBUF via pkg-config by default; override with PROTOBUF_CFLAGS /
#                 PROTOBUF_LIBS if pkg-config can't find it.
#
# Example against a CMake build tree at build/:
#   ONNX_INCLUDE="third_party/onnx:build" \
#   ONNX_LIB="build:build/lib" \
#   bench/graph_shape_inference_teardown_bench.sh 20000 3 1 2 4 1
#
# Args are forwarded to the binary:
#   [iters] [num_inputs] [num_outputs] [num_attrs] [rank] [num_const_inputs]
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="$here/graph_shape_inference_teardown_bench.cpp"
bin="${BENCH_BIN:-/tmp/graph_shape_inference_teardown_bench}"

# protobuf flags (pkg-config unless overridden).
if [[ -z "${PROTOBUF_CFLAGS:-}" || -z "${PROTOBUF_LIBS:-}" ]]; then
  if pkg-config --exists protobuf 2>/dev/null; then
    PROTOBUF_CFLAGS="${PROTOBUF_CFLAGS:-$(pkg-config --cflags protobuf)}"
    PROTOBUF_LIBS="${PROTOBUF_LIBS:-$(pkg-config --libs protobuf)}"
  else
    echo "protobuf not found via pkg-config; set PROTOBUF_CFLAGS/PROTOBUF_LIBS" >&2
    PROTOBUF_CFLAGS="${PROTOBUF_CFLAGS:-}"
    PROTOBUF_LIBS="${PROTOBUF_LIBS:--lprotobuf}"
  fi
fi

inc_flags=()
IFS=':' read -ra incs <<< "${ONNX_INCLUDE:-}"
for d in "${incs[@]}"; do [[ -n "$d" ]] && inc_flags+=("-I$d"); done

lib_flags=()
IFS=':' read -ra libs <<< "${ONNX_LIB:-}"
for d in "${libs[@]}"; do [[ -n "$d" ]] && lib_flags+=("-L$d" "-Wl,-rpath,$d"); done

set -x
g++ -O2 -std=c++17 "$src" -o "$bin" \
  "${inc_flags[@]}" ${PROTOBUF_CFLAGS} \
  "${lib_flags[@]}" -lonnx -lonnx_proto ${PROTOBUF_LIBS}
set +x

"$bin" "$@"
