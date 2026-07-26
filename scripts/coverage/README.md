# Combined C++ + Python coverage

onnxsim is a hybrid project. The C++ in `onnxsim/*.cpp` / `*.cc` is compiled
into the `onnxsim_cpp2py_export` extension, and the Python in `onnxsim/*.py`
drives it. A single `pytest` run exercises both halves in the same process, so
both can be measured at once:

| Layer  | Tool                    | How                                                                 |
| ------ | ----------------------- | ------------------------------------------------------------------- |
| Python | `coverage.py` (`pytest-cov`) | instruments the `.py` modules directly                         |
| C++    | `gcov` + `gcovr`        | the extension is built with `--coverage`; running it emits `.gcda` profiles that `gcovr` reports on |

## Quick start

```bash
python3 -m pip install pytest pytest-cov gcovr
scripts/coverage/run_coverage.sh
```

This builds an instrumented editable install, runs the test suite, and writes
reports to `./coverage-report/`:

```
coverage-report/python.xml       Cobertura XML  (Python)
coverage-report/python-html/     HTML           (Python)
coverage-report/cpp.xml          Cobertura XML  (C++)
coverage-report/cpp.html         HTML           (C++)
```

Both XML files are Cobertura, so a single Codecov/Coveralls/Sonar upload can
ingest them together for one unified C++/Python view.

Pass extra arguments straight through to pytest:

```bash
# only the fast, torch-free tests
scripts/coverage/run_coverage.sh tests/test_fusion_patterns.py tests/test_backend.py
```

## How it works

The mechanism is that the same C++ object code runs behind the Python
extension:

1. `COVERAGE=1` tells `setup.py` to configure CMake with `-DONNXSIM_COVERAGE=ON`
   (and force a `Debug`/`-O0` build for accurate line counts). The CMake option
   adds `--coverage` to onnxsim's own C++ targets, which emits `.gcno` files
   into `.setuptools-cmake-build/` at compile time.
2. When pytest imports `onnxsim` and calls into the extension, every C++ line
   that runs writes a `.gcda` profile next to its `.gcno`.
3. `coverage.py` records the Python side of the same run.
4. `gcovr` reads the `.gcda`/`.gcno` pairs and produces the C++ report;
   `coverage.py` produces the Python report.

## Environment overrides

| Variable          | Default             | Meaning                                            |
| ----------------- | ------------------- | -------------------------------------------------- |
| `COV_OUTPUT_DIR`  | `./coverage-report` | where reports are written                          |
| `COV_SKIP_BUILD`  | unset               | set to `1` to reuse an existing coverage build     |
| `COV_PYTEST_ARGS` | `tests`             | default pytest args when none are passed on the CLI |

## Doing it by hand

The script is a thin wrapper; the manual flow is:

```bash
# 1. instrumented editable build
COVERAGE=1 python3 -m pip install -e . --no-build-isolation

# 2. clear stale C++ profiles, then run the suite under coverage.py
find .setuptools-cmake-build -name '*.gcda' -delete
python3 -m pytest --cov=onnxsim --cov-report=term-missing tests

# 3. collect the C++ side
gcovr --root . --filter onnxsim/ --exclude '.*third_party.*' \
      --print-summary --html-details coverage-report/cpp.html \
      .setuptools-cmake-build
```

## Requirements

- A GCC- or Clang-compatible toolchain (MSVC has no `--coverage`), plus `cmake`
  and `ninja`.
- `gcovr`'s bundled `gcov` must match the compiler used for the build. With GCC
  this is automatic; with Clang, point gcovr at `llvm-cov`:
  `gcovr --gcov-executable "llvm-cov gcov" ...`.
