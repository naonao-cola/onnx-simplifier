#!/usr/bin/env bash
#
# Collect combined C++ + Python code coverage for onnxsim from a single pytest
# run.
#
# onnxsim is a hybrid project: the C++ in onnxsim/*.cpp / *.cc is compiled into
# the `onnxsim_cpp2py_export` extension, and the Python in onnxsim/*.py drives
# it. A pytest run exercises both halves in one process, so we can measure both
# at once:
#
#   * Python -> coverage.py (pytest-cov) instruments the .py modules.
#   * C++    -> the extension is built with --coverage (gcov). Every C++ line
#              that runs behind the extension writes a .gcda profile into the
#              build directory, which gcovr turns into a report afterwards.
#
# Reports land in a coverage output directory (default: ./coverage-report):
#
#   coverage-report/python.xml   Cobertura XML  (Python, coverage.py)
#   coverage-report/python-html/ HTML           (Python)
#   coverage-report/cpp.xml      Cobertura XML  (C++, gcovr)
#   coverage-report/cpp.html     HTML           (C++, gcovr --html-details)
#
# Both XML files are Cobertura, so a single upload step (Codecov, Coveralls,
# Sonar, ...) can ingest them together for a unified C++/Python view.
#
# Usage:
#   scripts/coverage/run_coverage.sh [pytest args...]
#
# Environment overrides:
#   COV_OUTPUT_DIR   where reports are written        (default: ./coverage-report)
#   COV_SKIP_BUILD   set to 1 to reuse an existing coverage build
#   COV_PYTEST_ARGS  default pytest args when none are passed on the CLI
#                    (default: "tests")
#
# Prerequisites: a GCC/Clang toolchain, cmake, ninja, and the Python packages
#   pytest pytest-cov gcovr
# plus whatever the tests themselves import (onnxruntime, torch, ...).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${REPO_ROOT}/.setuptools-cmake-build"
OUTPUT_DIR="${COV_OUTPUT_DIR:-${REPO_ROOT}/coverage-report}"
PYTEST_ARGS=("$@")
if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then
  # shellcheck disable=SC2206  # intentional word splitting of default args
  PYTEST_ARGS=(${COV_PYTEST_ARGS:-tests})
fi

echo "==> onnxsim combined C++/Python coverage"
echo "    repo:    ${REPO_ROOT}"
echo "    output:  ${OUTPUT_DIR}"
echo "    pytest:  ${PYTEST_ARGS[*]}"

# 1. Build the extension with coverage instrumentation (unless reusing a build).
#    COVERAGE=1 makes setup.py add -DONNXSIM_COVERAGE=ON and force a Debug/-O0
#    build so line counts are accurate. An editable install keeps the .gcno
#    files (and the source paths baked into them) in .setuptools-cmake-build.
if [ "${COV_SKIP_BUILD:-0}" != "1" ]; then
  echo "==> Building instrumented extension (COVERAGE=1, editable install)"
  COVERAGE=1 CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}" \
    python3 -m pip install -e . --no-build-isolation
else
  echo "==> COV_SKIP_BUILD=1: reusing existing build in ${BUILD_DIR}"
fi

if [ ! -d "${BUILD_DIR}" ]; then
  echo "error: build directory ${BUILD_DIR} not found; run without COV_SKIP_BUILD" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# 2. Zero out any stale C++ profiles so this run's numbers are clean.
echo "==> Clearing stale .gcda profiles"
find "${BUILD_DIR}" -name '*.gcda' -delete 2>/dev/null || true

# 3. Run pytest under coverage.py. `--cov=onnxsim` measures the Python package;
#    the C++ profiles accumulate as a side effect of the extension running.
echo "==> Running pytest with Python coverage"
python3 -m pytest \
  --cov=onnxsim \
  --cov-report="xml:${OUTPUT_DIR}/python.xml" \
  --cov-report="html:${OUTPUT_DIR}/python-html" \
  --cov-report=term-missing \
  "${PYTEST_ARGS[@]}"

# 4. Turn the accumulated .gcda profiles into C++ coverage reports. --root keeps
#    paths repo-relative; the filters restrict the report to onnxsim's own C++
#    (dropping vendored ONNX / onnxruntime and system headers).
#
#    --gcov-ignore-parse-errors=suspicious_hits.warn: a hot loop exercised
#    across the whole suite can accumulate a hit count so large (billions) that
#    gcov emits a value gcovr 8.x flags as "suspicious" and, by default, aborts
#    the whole run over (exit 64). Those counts are legitimate, so downgrade the
#    suspicious-hit error to a warning and keep going.
echo "==> Collecting C++ coverage with gcovr"
gcovr \
  --root "${REPO_ROOT}" \
  --filter "${REPO_ROOT}/onnxsim/" \
  --exclude '.*third_party.*' \
  --gcov-ignore-parse-errors=suspicious_hits.warn \
  --print-summary \
  --xml-pretty --output "${OUTPUT_DIR}/cpp.xml" \
  --html-details "${OUTPUT_DIR}/cpp.html" \
  "${BUILD_DIR}"

echo ""
echo "==> Done. Reports written to ${OUTPUT_DIR}:"
echo "    Python: ${OUTPUT_DIR}/python.xml  +  ${OUTPUT_DIR}/python-html/index.html"
echo "    C++:    ${OUTPUT_DIR}/cpp.xml      +  ${OUTPUT_DIR}/cpp.html"
