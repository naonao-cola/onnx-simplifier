# Combined C++ + Python + JS + Rust coverage

onnxsim is a hybrid project. The C++ in `onnxsim/*.cpp` / `*.cc` is compiled
into the `onnxsim_cpp2py_export` extension, and the Python in `onnxsim/*.py`
drives it. A single `pytest` run exercises both halves in the same process, so
both can be measured at once. The browser converter under
`scripts/convertmodel/` adds a third layer of JavaScript, covered separately by
its own Node test suite, and the Rust wrapper crates under `rust/` add a fourth
(see the Rust section below):

| Layer  | Tool                    | How                                                                 |
| ------ | ----------------------- | ------------------------------------------------------------------- |
| Python | `coverage.py` (`pytest-cov`) | instruments the `.py` modules directly                         |
| C++    | `gcov` + `gcovr`        | the extension is built with `--coverage`; running it emits `.gcda` profiles that `gcovr` reports on |
| JS     | `c8` (V8 coverage)      | runs the convertmodel Node unit tests and reports on the `.mjs` modules they exercise |

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

Both XML files are Cobertura, which most coverage viewers accept. In CI these
are combined with the JS and Rust reports into a single summary and pull-request
comment; see "CI structure" below for how that is wired up.

## JavaScript (convertmodel)

`run_coverage.sh` covers only the C++/Python halves. The JS coverage is a
separate, self-contained step because it needs a Node toolchain rather than the
C++/Python build:

```bash
cd scripts/convertmodel
npm ci
npm run coverage            # c8 wraps `npm run test:all`
```

This runs every Node unit test (`test/*.test.mjs`) under
[`c8`](https://github.com/bcoe/c8) and writes `text`, `html`, and `cobertura`
reports to `scripts/convertmodel/coverage/`:

```
scripts/convertmodel/coverage/cobertura-coverage.xml   Cobertura XML  (JS)
scripts/convertmodel/coverage/index.html               HTML           (JS)
```

Scope and reporters live in `scripts/convertmodel/.c8rc.json`. `all: true` plus
the `include` list report every node-testable module even when a given run does
not import it, so the number reflects the whole unit-tested surface rather than
just the files loaded this run. The browser-only glue (`hf_load.mjs`,
`*_view.mjs`, `inference_browser.mjs`, `worker.js`) drives the DOM and has no
unit test, so it is intentionally left out.

The CI `cpp-python-js` job copies `cobertura-coverage.xml` to
`coverage-report/js.xml`, so JS lands in the one combined table alongside C++,
Python and Rust.

## Rust (bindings)

The Rust wrapper crates under `rust/` are covered in a separate CI job because
they need a linked native `onnxsim_c` library rather than the C++/Python/Node
toolchains above. Coverage there uses
[`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov):

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
cd rust
# Prebuilt ONNX Runtime keeps the required native build fast.
ONNXSIM_PREBUILT_ORT=1 cargo llvm-cov --workspace --cobertura \
  --output-path ../coverage-report/rust.xml
```

`cargo-llvm-cov` instruments only the wrapper crates (`onnxsim`, `onnxsim-sys`);
the C++ core they call into is already covered by the C++ report above. Like the
other reports, the Rust output is Cobertura XML.

Branch coverage needs the nightly toolchain (`cargo +nightly llvm-cov --branch`,
which sets `-Zcoverage-options=branch`); on stable only region/line/function
coverage is reported. CI runs this job on nightly to fill the branch column and
excludes `onnxsim-sys/build.rs` (nightly instruments the build script, whose
"coverage" reflects the build path taken, not the tests). See `rust/README.md`.

## CI structure (one combined comment)

In CI (`.github/workflows/coverage.yml`) each language is measured in its own
job — `cpp-python-js` (this script + the Node suite) and `rust`
(`cargo-llvm-cov`) — and each uploads its Cobertura XML as an artifact. A final
`report` job downloads them all and feeds the comma-separated list
(`cpp.xml,python.xml,js.xml,rust.xml`) to a single
[`irongut/CodeCoverageSummary`](https://github.com/irongut/CodeCoverageSummary)
run, which writes the combined table to the Actions job summary and posts it as
one sticky pull-request comment via
[`marocchino/sticky-pull-request-comment`](https://github.com/marocchino/sticky-pull-request-comment)
— no external service or secret required. Splitting measurement across jobs but
downloading the artifacts into one `report` job keeps every language in a single
comment. The same Cobertura files can just as easily feed Codecov, Coveralls,
SonarQube, or GitHub's native Code Quality coverage.

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
#    (--cov-branch so the Python side reports branch coverage, not just lines)
find .setuptools-cmake-build -name '*.gcda' -delete
python3 -m pytest --cov=onnxsim --cov-branch --cov-report=term-missing tests

# 3. collect the C++ side
gcovr --root . --filter onnxsim/ --exclude '.*third_party.*' \
      --gcov-ignore-parse-errors=suspicious_hits.warn \
      --print-summary --html-details coverage-report/cpp.html \
      .setuptools-cmake-build
```

> `--gcov-ignore-parse-errors=suspicious_hits.warn` keeps gcovr from aborting
> when a hot loop's hit count grows large enough (billions) that gcov emits a
> value gcovr 8.x flags as "suspicious". Those counts are real; the flag
> downgrades them from a fatal error to a warning.

## Requirements

- A GCC- or Clang-compatible toolchain (MSVC has no `--coverage`), plus `cmake`
  and `ninja`.
- `gcovr`'s bundled `gcov` must match the compiler used for the build. With GCC
  this is automatic; with Clang, point gcovr at `llvm-cov`:
  `gcovr --gcov-executable "llvm-cov gcov" ...`.
