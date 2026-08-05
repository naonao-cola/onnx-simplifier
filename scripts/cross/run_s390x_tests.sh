#!/usr/bin/env bash
# Install the cross-built onnxsim into the s390x rootfs and run the test suite
# there under qemu-user, i.e. big endian.
#
# Prerequisites:
#   scripts/cross/bootstrap_s390x_rootfs.sh   # rootfs + binfmt
#   scripts/cross/build_s390x_extension.sh    # the extension itself
#
# Set SYSROOT=/rootfs-amd64 and BUILD=.native-build-control/onnxsim-build to run
# the identical stack little-endian as a control (see build_native_control.sh);
# the two runs differ only in byte order, so any difference is an endianness bug.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSROOT="${SYSROOT:-/rootfs-s390x}"
BUILD="${BUILD:-${REPO_ROOT}/.cross-build-s390x/onnxsim-build}"

ONNX_SRC="${REPO_ROOT}/third_party/onnx-optimizer/third_party/onnx"
ONNX_BUILD="${BUILD}/third_party/onnx-optimizer/third_party/onnx"
DIST="${SYSROOT}/usr/lib/python3/dist-packages"
PKG="${DIST}/onnxsim"
LIBS="${SYSROOT}/work/pylibs"

SO="$(find "${BUILD}" -maxdepth 1 -name 'onnxsim_cpp2py_export*.so' -print -quit)"
[[ -n "${SO}" ]] || { echo "extension not built; run build_s390x_extension.sh first"; exit 1; }

# ---------------------------------------------------------------------------
# onnxsim itself: the pure-Python package plus the built extension.
# ---------------------------------------------------------------------------
echo "== installing onnxsim into ${SYSROOT} =="
rm -rf "${PKG}"; mkdir -p "${PKG}"
find "${REPO_ROOT}/onnxsim" -maxdepth 1 -name '*.py' -exec cp {} "${PKG}/" \;
# setup.py normally generates this; the cross-build never runs setup.py.
cat > "${PKG}/version.py" <<EOF
version = '$(cat "${REPO_ROOT}/VERSION")'
git_version = None
dev_count = None
EOF
cp "${SO}" "${PKG}/onnxsim_cpp2py_export.abi3.so"

# ---------------------------------------------------------------------------
# The vendored onnx, assembled from the same build. The distro's python3-onnx
# is much older than what onnxsim's C++ is compiled against and fails a large
# part of the suite for reasons unrelated to byte order, so put the matching
# version ahead of it on PYTHONPATH. protobuf ships a pure-Python wheel, which
# is architecture independent and so needs no s390x build.
# ---------------------------------------------------------------------------
echo "== installing vendored onnx $(cat "${ONNX_SRC}/VERSION_NUMBER") =="
rm -rf "${LIBS}"; mkdir -p "${LIBS}"
> "${LIBS}/.keep"
python3 -m pip install --quiet --target="${LIBS}" --no-deps \
  "$(grep -oE '"protobuf>=[0-9.]+"' "${ONNX_SRC}/pyproject.toml" | tr -d '"')" \
  "$(grep -oE '"typing_extensions>=[0-9.]+"' "${ONNX_SRC}/pyproject.toml" | tr -d '"')"

cp -r "${ONNX_SRC}/onnx" "${LIBS}/onnx"
rm -rf "${LIBS}/onnx/test" "${LIBS}/onnx/backend/test/data"
cp "${ONNX_BUILD}"/onnx/*_pb.py "${ONNX_BUILD}"/onnx/*_pb2.py "${LIBS}/onnx/"
cp "${ONNX_BUILD}"/onnx_cpp2py_export*.so "${LIBS}/onnx/"
# onnx.__init__ reads its version through importlib.metadata.
ONNX_VER="$(cat "${ONNX_SRC}/VERSION_NUMBER")"
mkdir -p "${LIBS}/onnx-${ONNX_VER}.dist-info"
printf 'Metadata-Version: 2.1\nName: onnx\nVersion: %s\n' "${ONNX_VER}" \
  > "${LIBS}/onnx-${ONNX_VER}.dist-info/METADATA"
: > "${LIBS}/onnx-${ONNX_VER}.dist-info/RECORD"

mkdir -p "${SYSROOT}/work"
rm -rf "${SYSROOT}/work/tests"
cp -r "${REPO_ROOT}/tests" "${SYSROOT}/work/tests"
cp "${REPO_ROOT}/pyproject.toml" "${SYSROOT}/work/"

echo "== environment =="
chroot "${SYSROOT}" /bin/sh -c 'cd /work && PYTHONPATH=/work/pylibs python3 -c "
import sys, numpy, onnx, onnxsim
print(sys.byteorder, \"endian | python\", sys.version.split()[0],
      \"| numpy\", numpy.__version__, \"| onnx\", onnx.__version__,
      \"| onnxsim\", onnxsim.__version__)
"'

# torch, timm and onnxruntime have no s390x builds, so the test modules that
# import them at module scope cannot be collected here. test_qnn_compat needs a
# model fixture that is not vendored.
echo "== pytest =="
exec chroot "${SYSROOT}" /bin/sh -c "cd /work && PYTHONPATH=/work/pylibs python3 -m pytest tests -p no:cacheprovider \
  --ignore=tests/test_mnn_llm_export.py --ignore=tests/test_python_api.py --ignore=tests/test_rfdetr.py \
  --ignore=tests/test_simple.py --ignore=tests/test_timm.py --ignore=tests/test_yolo.py \
  --ignore=tests/test_qnn_compat.py --ignore=tests/test_modelopt_integration.py ${PYTEST_ARGS:-}"
