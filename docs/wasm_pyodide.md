# Pyodide / wasm32-emscripten Python extension (experimental)

**Status: experimental / build-verified only, NOT runtime-verified.** Nobody
has loaded the module this produces inside a real Pyodide/JS runtime and
confirmed `import onnxsim` actually works -- only structural checks have been
done (it's a genuine WebAssembly binary, it has a `dylink.0` section, it
exports `PyInit_onnxsim_cpp2py_export`, and it has the shape of imports a
Pyodide side module needs resolved at `dlopen` time). Treat it as a
promising, structurally-correct artifact, not a working package.

## What this is

`build_wasm_pyodide.sh` cross-compiles onnxsim's nanobind Python extension
(`onnxsim_cpp2py_export`, the module `import onnxsim` loads) for
`wasm32-emscripten`, producing `onnxsim_cpp2py_export.abi3.so` as a Pyodide
**side module** -- the format `pyodide.loadPackage` / `micropip` expect for a
compiled extension: a relocatable WebAssembly binary that Pyodide's own
runtime `dlopen`s and links against the main Pyodide module at import time,
rather than a self-contained standalone `.wasm`.

This is a **third** build path, separate from and not touching:

- **The Python wheel build** (`setup.py`) -- builds the same extension for
  the *host* platform, always with `ONNXSIM_BUILTIN_ORT=OFF`. See
  `CLAUDE.md` for why this never compiles ONNX Runtime.
- **The standalone WASM CLI build** (`build_wasm.sh`) -- builds
  `onnxsim_bin`, a self-contained CLI/JS module for the browser, with
  `ONNXSIM_PYTHON=OFF`.

Pyodide needs neither of those: it needs the *Python extension*, cross
compiled for wasm32, in Pyodide's own module format. Hence a new script
rather than a flag on either existing one.

## Why this needs a script at all

Four separate obstacles had to be worked around to get from "onnxsim builds
for the host" to a linkable wasm side module. None of them are onnxsim bugs
in the usual sense -- they're toolchain/version-matching issues specific to
cross-compiling a CPython C extension with CMake under Emscripten.

### 1. protoc/protobuf version matching

Same requirement as `build_wasm.sh`'s existing protoc handling (see that
script and `docs/wasm_ort_web.md`'s protobuf section): CMake's
`ONNX_CUSTOM_PROTOC_EXECUTABLE` needs a **host** protoc whose codegen matches
the vendored protobuf runtime being linked, or generated `.pb.*` code
mismatches the runtime headers. This script reuses `build_wasm.sh`'s exact
protoc discovery/build logic (`third_party/onnx/workflow_scripts/protobuf/build_protobuf_unix.sh`)
rather than duplicating or diverging from it.

### 2. Plumbing target Python headers through three separate CMake hint namespaces

`ONNXSIM_PYTHON=ON` triggers Python detection in onnxsim's own
`CMakeLists.txt`, onnx's, onnx-optimizer's, and nanobind's own CMake config --
and CMake's `FindPython`, `FindPython3`, and nanobind's config each read
hints from a **different** variable namespace, so missing any one of them
fails outright even with the other two correctly set:

- onnx's own `CMakeLists.txt` calls the **versioned** `find_package(Python3
  ...)`, correctly split into `Interpreter`/`Development` -- hints:
  `Python3_EXECUTABLE`, `Python3_INCLUDE_DIR(S)`.
- onnx-optimizer's `CMakeLists.txt` instead calls the **generic, unversioned**
  `find_package(Python 3 REQUIRED COMPONENTS Interpreter
  Development.Module)` -- a separate CMake module with its own variables:
  `Python_EXECUTABLE`, `Python_INCLUDE_DIR(S)`. Passing only the `Python3_*`
  hints leaves this call unable to find `Python.Development` at all
  (`Could NOT find Python (missing: Python_INCLUDE_DIRS
  Development.Module)`) -- this only surfaces on a fresh configure with no
  stale `CMakeCache.txt` from a previous, differently-argued run to paper
  over it, which is how it slipped past local testing before CI caught it.
- nanobind's own CMake config reads the undocumented plural
  `Python_INCLUDE_DIRS` variable specifically (not `Python3_INCLUDE_DIRS`).

So both the versioned and unversioned forms of every hint need to be passed,
to the same target headers, together with:

- Host and target Python **minor versions must match** (e.g. host
  `python3.14` for a Pyodide release shipping CPython 3.14.x), since
  `Python3_EXECUTABLE`/`Python_EXECUTABLE` point at the *host* interpreter
  CMake actually runs to introspect Python, while the `INCLUDE_DIR(S)` hints
  point at the *target*'s headers.
- `Python3_FIND_ABI`/`Python_FIND_ABI` set to `ANY;ANY;ANY;ANY`, to avoid a
  strict-ABI mismatch rejection between the host interpreter and the
  target's ABI tag.

### 3. nanobind 3.0.0's missing `<cstdio>` include

`nb_backend.h` / `nb_types.h` / `ndarray.h` use `stderr`/`fprintf` without
including `<cstdio>`. This works by accident on glibc hosts, which pull it in
transitively through some other header, and fails under Emscripten's libc++,
which doesn't. This is a real upstream nanobind bug worth reporting there --
not an onnxsim-specific issue. Worked around here with `-include cstdio` on
`CMAKE_CXX_FLAGS` for this build only.

### 4. Emscripten's CMake toolchain disables native MODULE/SHARED support

Emscripten's own CMake toolchain file
(`$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`)
hardcodes `set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)`.
By design, CMake cannot produce a `MODULE`/`SHARED` library under this
toolchain -- the `onnxsim_cpp2py_export` target (a nanobind extension module,
normally `MODULE`) silently falls back to being archived as a **static**
library containing only its own single object file
(`cpp2py_export.cc.o`). This is also why Pyodide's own `pyodide build`
bypasses CMake/setuptools' dynamic-linking support entirely, via a
compiler-wrapping shim (`pywasmcross`), instead of relying on it.

The fix -- and the reason this script exists rather than just being a CMake
invocation -- is a **manual link step outside CMake**, after the CMake build
completes, using the exact object file and dependency archives CMake already
produced:

```sh
em++ -O2 -sSIDE_MODULE=2 -sEXPORTED_FUNCTIONS=_PyInit_onnxsim_cpp2py_export \
  -o onnxsim_cpp2py_export.abi3.so \
  CMakeFiles/onnxsim_cpp2py_export.dir/onnxsim/cpp2py_export.cc.o \
  <every *.a found under the build dir: abseil, protobuf, onnx,
   onnx-optimizer, onnxsim's own static libs, nanobind-static-abi3, ...>
```

`-sEXPORTED_FUNCTIONS=_PyInit_onnxsim_cpp2py_export` is required: without an
explicit root symbol, the linker's dead-code elimination strips the module
down to a couple KB, since nothing in this standalone link otherwise
references the nanobind entry point (there's no `main()` or other symbol
pulling it in, unlike inside Pyodide's real loader). The script discovers the
exact entry symbol from the object file via `llvm-nm` rather than assuming it
matches the module name, and discovers the archive list by scanning the build
directory for `*.a` at run time -- **not** a hand-maintained list, which
would silently go stale as dependencies change.

## Building

```sh
# Activate an emsdk whose Emscripten version compiles onnxsim's vendored
# protobuf cleanly. pyodide-build's own default (Emscripten 3.1.46 /
# clang-18) fails on a protobuf `constinit` compile error; a newer
# Emscripten (e.g. the one bundled with a recent Pyodide xbuildenv, such as
# Pyodide 314.0.5's Emscripten 5.0.3) is known to work. Use the SAME emsdk
# for the whole run -- see the script's own header comment for why.
source /path/to/matching/emsdk/emsdk_env.sh

PYODIDE_PYTHON_INCLUDE=/path/to/xbuildenv/.../include/python3.14 \
PYTHON_EXECUTABLE=/path/to/host/python3.14 \
  ./build_wasm_pyodide.sh
```

`PYTHON_EXECUTABLE` needs `nanobind` pip-installed (`pip install nanobind`);
CMake shells out to `python -m nanobind --cmake_dir` on it to locate
nanobind's CMake package. `PYODIDE_PYTHON_INCLUDE` is the target Pyodide
release's Python headers dir, e.g. from a Pyodide xbuildenv
(`pyodide xbuildenv install <version>`, or reuse one already downloaded to
`~/.cache/pyodide-build`). Both are asserted with a clear error if unset. See
the script's own header comment for the full prerequisite list and defaults
(`BUILD_DIR` defaults to `build-wasm-pyodide`, reused across reruns like
`build_wasm.sh`'s `BUILD_DIR`).

Output: `<BUILD_DIR>/onnxsim_cpp2py_export.abi3.so`, the script's last line.

## What's next

CI now attempts the runtime verification this section used to describe as
unscripted: `.github/workflows/pyodide-wasm.yml` runs `build_wasm_pyodide.sh`
end to end (pinned Emscripten 5.0.3, Pyodide 314.0.5 xbuildenv headers), then
loads the resulting `.abi3.so` inside a real Pyodide runtime (the `pyodide`
npm package, under Node -- `scripts/pyodide_smoke_test.mjs`) and confirms
`import onnxsim_cpp2py_export` succeeds and a real binding
(`_list_optimizers()`) runs. That's still narrower than the top of this doc
describes: it exercises the low-level nanobind extension module alone, not
`import onnxsim` / `onnxsim.simplify()`, which additionally needs `onnx` and
`rich` present inside the Pyodide environment -- not yet scripted.

**Until that workflow has actually gone green on a real run, treat this as
still unverified** -- the "Status" line at the top of this doc is the source
of truth; update it only once CI confirms the import works, not based on this
section alone.
