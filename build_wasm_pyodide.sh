#!/usr/bin/env bash
set -ex

# Cross-compiles onnxsim's Python C-extension (the nanobind module
# `onnxsim_cpp2py_export`) for Pyodide / wasm32-emscripten, as a Pyodide
# "side module" (.abi3.so). See docs/wasm_pyodide.md for the full story,
# including why several manual steps below are needed and what is/isn't
# verified so far.
#
# This is a THIRD build path, separate from and not touching:
#   - the Python wheel build (setup.py, always ONNXSIM_BUILTIN_ORT=OFF,
#     native host target)
#   - the standalone WASM CLI build (build_wasm.sh, ONNXSIM_PYTHON=OFF,
#     ONNXSIM_BUILTIN_ORT=ON by default)
# (see CLAUDE.md). Opt-in and additive only.
#
# ---------------------------------------------------------------------------
# Prerequisites (this script does not install any of these):
#
#   - emcmake/em++ on PATH, from an emsdk whose Emscripten version is
#     KNOWN GOOD for onnxsim's vendored protobuf (clang-18/Emscripten
#     3.1.46, pyodide-build's own default, fails on a protobuf `constinit`
#     compile error -- any newer Emscripten avoids it). Activate it the same
#     way build_wasm.sh expects emcmake to already be on PATH, e.g.:
#       source /path/to/emsdk/emsdk_env.sh
#     Use the SAME emsdk for the whole script run -- the manual link step
#     near the end re-invokes em++ standalone, and mixing two different
#     Emscripten toolchains between the CMake compile and that link step
#     is not a combination this script tests for.
#
#     RECOMMENDED: Emscripten 4.0.9, matching Pyodide release 0.29.4 below.
#     A newer Emscripten (e.g. 5.0.3, matching Pyodide 314.0.5) compiles
#     fine too, but produces a module with a DIFFERENT, incompatible ABI
#     epoch than PyPI's only `onnx` wheel -- see the Pyodide release note
#     just below and docs/wasm_pyodide.md for why that's the pin that
#     actually matters, not just "new enough to compile".
#
#   - PYODIDE_PYTHON_INCLUDE: the TARGET Python headers directory, i.e. the
#     cross headers for the Python version the target Pyodide release ships,
#     e.g. from a Pyodide xbuildenv:
#       .../xbuildenv/pyodide-root/cpython/installs/python-3.13.2/include/python3.13
#     Get one via `pyodide xbuildenv install <version>` (pyodide-build) or
#     reuse an already-downloaded one from ~/.cache/pyodide-build.
#
#     RECOMMENDED: Pyodide release 0.29.4. Its ABI epoch
#     (`sysconfig.get_config_var("PYODIDE_ABI_VERSION")` -> `2025_0`) is the
#     SAME epoch PyPI's `onnx` wheel is tagged for
#     (`onnx-*-pyemscripten_2025_0_wasm32.whl`) -- confirmed by loading
#     onnx's real compiled extension in this exact runtime. Building against
#     a later Pyodide release (a different epoch) still produces a working
#     onnxsim_cpp2py_export.abi3.so, but one that can't be combined with
#     onnx's only published wasm wheel to run full `onnxsim.simplify()`.
#
#   - PYTHON_EXECUTABLE (optional, defaults to `python3` on PATH): a HOST
#     Python interpreter with `nanobind` pip-installed (`pip install
#     nanobind`) -- CMake shells out to `python -m nanobind --cmake_dir` on
#     this interpreter to locate nanobind's CMake package config. Its MINOR
#     version should match the target Python above (e.g. host python3.13 for
#     Pyodide 0.29.4's CPython 3.13.x); CMake's Python3 find logic
#     in onnx-optimizer's CMakeLists.txt runs this interpreter directly and
#     can fail on a minor-version mismatch against the target headers.
#
#   - BUILD_DIR (optional, defaults to build-wasm-pyodide): where to
#     configure/build. Reused across reruns like build_wasm.sh's BUILD_DIR.
#
# Usage:
#   source /path/to/matching/emsdk/emsdk_env.sh
#   PYODIDE_PYTHON_INCLUDE=/path/to/target/include/python3.13 \
#   PYTHON_EXECUTABLE=/path/to/host/python3.13 \
#     ./build_wasm_pyodide.sh
# ---------------------------------------------------------------------------

# Check the toolchain is available before anything else, same as build_wasm.sh.
command -v emcmake
command -v em++

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ -z "${PYODIDE_PYTHON_INCLUDE:-}" ]; then
    echo "error: PYODIDE_PYTHON_INCLUDE must be set to the target Pyodide" >&2
    echo "Python headers directory. See the comment block at the top of" >&2
    echo "this script for details." >&2
    exit 1
fi
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}
# Resolve to an absolute path before handing it to CMake. A bare name here
# left CMake's own find_program() to re-resolve "python3", which -- only
# intermittently, same toolchain/versions, fresh checkout each time -- picked
# a DIFFERENT, wrong-architecture interpreter (e.g. the runner's native
# /bin/python) for onnx-optimizer's generic find_package(Python ...) call
# (see docs/wasm_pyodide.md's "Plumbing target Python headers" section),
# failing the whole configure with "Wrong architecture for the interpreter".
# An already-absolute, already-verified-to-exist path removes that
# re-resolution step entirely: CMake takes it as-is.
if command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1; then
    PYTHON_EXECUTABLE=$(command -v "$PYTHON_EXECUTABLE")
else
    echo "error: PYTHON_EXECUTABLE '$PYTHON_EXECUTABLE' not found" >&2
    exit 1
fi

# --- Host protoc: identical approach to build_wasm.sh -- reused verbatim so
# the two scripts don't drift on how a matching host protoc is found/built.
# A mismatched protoc's codegen doesn't match the vendored protobuf runtime
# (see docs/wasm_pyodide.md).
if which protoc ; then
    PROTOC=$(which protoc)
else
    . ./third_party/onnx/workflow_scripts/protobuf/build_protobuf_unix.sh $(nproc) $PWD/protobuf
    PROTOC=$(which protoc)
fi

set -u -o pipefail

BUILD_DIR=${BUILD_DIR:-build-wasm-pyodide}
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake's FindPython, FindPython3, and nanobind's own config each read from
# a DIFFERENT hint-variable namespace, so all three need to be seeded with
# the same values or whichever one is missed fails outright:
#   - onnx's own CMakeLists.txt calls find_package(Python3 ...) (versioned)
#     -> Python3_EXECUTABLE / Python3_INCLUDE_DIR(S).
#   - onnx-optimizer's CMakeLists.txt instead calls the GENERIC, UNVERSIONED
#     find_package(Python 3 ...) -> Python_EXECUTABLE / Python_INCLUDE_DIR,
#     a separate CMake module with its own variables; passing only the
#     Python3_* ones leaves this one unable to find Python.Development at
#     all ("Could NOT find Python (missing: Python_INCLUDE_DIRS
#     Development.Module)").
#   - nanobind's own CMake config reads the undocumented plural
#     "Python_INCLUDE_DIRS" variable specifically.
# See docs/wasm_pyodide.md.
emcmake cmake \
    -DONNX_CUSTOM_PROTOC_EXECUTABLE="$PROTOC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-include cstdio" \
    -DONNXSIM_PYTHON=ON \
    -DONNXSIM_BUILTIN_ORT=OFF \
    -DONNX_BUILD_PYTHON=ON \
    -DONNX_INSTALL=OFF \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython3_INCLUDE_DIR="$PYODIDE_PYTHON_INCLUDE" \
    -DPython3_INCLUDE_DIRS="$PYODIDE_PYTHON_INCLUDE" \
    -DPython3_FIND_ABI="ANY;ANY;ANY;ANY" \
    -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython_INCLUDE_DIR="$PYODIDE_PYTHON_INCLUDE" \
    -DPython_INCLUDE_DIRS="$PYODIDE_PYTHON_INCLUDE" \
    -DPython_FIND_ABI="ANY;ANY;ANY;ANY" \
    "$SCRIPT_DIR"

cmake --build . --target onnxsim_cpp2py_export -j"$(nproc)"

# --- Manual side-module link ------------------------------------------------
#
# Emscripten's own CMake toolchain file hardcodes
# TARGET_SUPPORTS_SHARED_LIBS=FALSE, so CMake's native MODULE/SHARED library
# support is unavailable under this toolchain by design (this is why
# Pyodide's own `pyodide build` bypasses CMake/setuptools' dynamic-linking
# support entirely via pywasmcross). The target above therefore built as a
# static ar archive containing only its own object file, not a loadable
# module -- link it into a real Pyodide side module by hand, reusing the
# exact object file and dependency archives CMake already produced.
MODULE_NAME=onnxsim_cpp2py_export

OBJ=$(find "CMakeFiles/${MODULE_NAME}.dir" -name 'cpp2py_export.cc.o' | head -1)
if [ -z "$OBJ" ]; then
    echo "error: could not find cpp2py_export.cc.o under CMakeFiles/${MODULE_NAME}.dir" >&2
    echo "(did the 'cmake --build' step above actually succeed?)" >&2
    exit 1
fi

# Every dependency archive CMake built in this tree: abseil, protobuf, onnx,
# onnx-optimizer, onnxsim's own static libs, nanobind-static-abi3, etc.
# Discovered at run time (not hand-maintained) so this never silently goes
# stale as dependencies are added/removed/renamed.
mapfile -t ARCHIVES < <(find . -name '*.a' | sort)
if [ "${#ARCHIVES[@]}" -eq 0 ]; then
    echo "error: no .a archives found under $PWD -- expected dependency" >&2
    echo "archives (abseil/protobuf/onnx/onnx-optimizer/onnxsim/nanobind)" >&2
    echo "to already be built at this point." >&2
    exit 1
fi

# nanobind's entry point follows CPython's extension-module ABI:
# PyInit_<module name>. Discover the exact symbol from the object file
# rather than assuming it matches MODULE_NAME, so a future module rename
# doesn't silently produce a link that's missing its only real entry point.
ENTRY_SYMBOL=""
if command -v llvm-nm >/dev/null 2>&1; then
    ENTRY_SYMBOL=$(llvm-nm "$OBJ" | awk '/ T PyInit_/ {print $NF; exit}')
fi
if [ -z "$ENTRY_SYMBOL" ]; then
    ENTRY_SYMBOL="PyInit_${MODULE_NAME}"
fi

OUT="${MODULE_NAME}.abi3.so"

# -sEXPORTED_FUNCTIONS is required: without it, the linker's dead-code
# elimination strips the module down to a couple KB, since nothing else in
# this standalone link references the nanobind entry point (there is no
# main()/other exported symbol pulling it in the way Pyodide's real module
# loader would).
em++ -O2 -sSIDE_MODULE=2 -sEXPORTED_FUNCTIONS="_${ENTRY_SYMBOL}" \
    -o "$OUT" \
    "$OBJ" \
    "${ARCHIVES[@]}"

OUT_PATH="$PWD/$OUT"
echo ""
echo "Built Pyodide side module: $OUT_PATH"
file "$OUT_PATH" || true
echo ""
echo "Next steps (NOT done by this script):"
echo "  - This has only been checked structurally (wasm binary, dylink.0"
echo "    section, exports ${ENTRY_SYMBOL}). It has NOT been loaded inside a"
echo "    real Pyodide/JS runtime -- 'import onnxsim' has not been confirmed"
echo "    to work. See docs/wasm_pyodide.md's Status section."
echo "  - To try it: place $OUT next to onnxsim's pure-Python package files"
echo "    on a Pyodide filesystem (e.g. loadPyodide() + pyodide.FS, or a"
echo "    micropip-installable wheel laid out the same way) and attempt"
echo "    'import onnxsim_cpp2py_export' / 'import onnxsim' from JS."
