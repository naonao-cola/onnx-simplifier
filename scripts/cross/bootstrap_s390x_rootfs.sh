#!/usr/bin/env bash
# Build an Ubuntu s390x (big endian) rootfs and register qemu-user for it, so
# onnxsim can be run and tested big-endian on an x86_64 host.
#
# s390x is the only mainstream big-endian Linux target a major distro still
# ships, which makes it the practical way to check onnxsim's byte-order
# assumptions. Run as root. See README.md for the full procedure.
set -euo pipefail

# ARCH=amd64 builds the little-endian control rootfs instead (same distro, same
# package versions), so a big-endian run can be diffed against one that differs
# only in byte order. Ubuntu keeps s390x on ports.ubuntu.com and amd64 on the
# main archive.
ARCH="${ARCH:-s390x}"
SUITE="${SUITE:-noble}"
if [[ "${ARCH}" == "amd64" ]]; then
  SYSROOT="${SYSROOT:-/rootfs-amd64}"
  MIRROR="${MIRROR:-http://archive.ubuntu.com/ubuntu/}"
else
  SYSROOT="${SYSROOT:-/rootfs-s390x}"
  MIRROR="${MIRROR:-http://ports.ubuntu.com/ubuntu-ports/}"
fi

command -v debootstrap >/dev/null || {
  echo "installing debootstrap + qemu-user-static + the s390x cross toolchain"
  apt-get update -q
  apt-get install -y -q debootstrap qemu-user-static g++-s390x-linux-gnu
}

# ---------------------------------------------------------------------------
# binfmt_misc so s390x binaries (including everything dpkg runs inside the
# chroot) transparently execute under qemu.
# ---------------------------------------------------------------------------
if [[ "${ARCH}" == "s390x" ]]; then
if [[ ! -e /proc/sys/fs/binfmt_misc/register ]]; then
  mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
fi
if [[ -e /proc/sys/fs/binfmt_misc/qemu-s390x ]]; then
  echo -1 > /proc/sys/fs/binfmt_misc/qemu-s390x
fi
# Bytes 0-6 are the ELF ident (64-bit, MSB, v1); 16-17 e_type; 18-19 e_machine
# (0x0016 = EM_S390). The mask zeroes EI_OSABI/EI_ABIVERSION -- glibc's own
# binaries (e.g. ldconfig.real) set EI_OSABI=GNU, and a mask that pins them to
# SYSV makes those, and only those, fail with "Exec format error" mid-install.
# The 0xfe on byte 17 accepts both ET_EXEC and ET_DYN (static-pie binaries).
printf ':qemu-s390x:M::\x7fELF\x02\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x16:\xff\xff\xff\xff\xff\xff\xff\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff:/usr/bin/qemu-s390x-static:F' \
  > /proc/sys/fs/binfmt_misc/register
fi

# ---------------------------------------------------------------------------
# The rootfs. python3-onnx/python3-numpy/python3-pytest exist for s390x in
# Ubuntu, which is what makes this practical -- PyPI has no s390x wheels.
# The distro onnx is older than the one onnxsim vendors; run_s390x_tests.sh
# installs the vendored version over it.
# ---------------------------------------------------------------------------
if [[ ! -e "${SYSROOT}/etc/os-release" ]]; then
  debootstrap --arch="${ARCH}" --variant=minbase --components=main,universe \
    --include=python3,python3-dev,python3-numpy,python3-onnx,python3-pytest,python3-protobuf,python3-pip,libpython3-dev,libprotobuf-dev,protobuf-compiler,ca-certificates,build-essential,cmake,git \
    "${SUITE}" "${SYSROOT}" "${MIRROR}"
fi

cat > "${SYSROOT}/etc/apt/sources.list" <<EOF
deb ${MIRROR} ${SUITE} main universe
deb ${MIRROR} ${SUITE}-updates main universe
EOF
cp /etc/resolv.conf "${SYSROOT}/etc/resolv.conf"
# Proxied environments hand pip a CA bundle by absolute path; make it resolve
# inside the chroot too.
if [[ -f /root/.ccr/ca-bundle.crt ]]; then
  mkdir -p "${SYSROOT}/root/.ccr"
  cp /root/.ccr/ca-bundle.crt "${SYSROOT}/root/.ccr/"
fi

# rich is a runtime dependency of onnxsim and pure Python, so the host's pip can
# drop it straight in. ml_dtypes is a C extension the vendored onnx needs and
# has no s390x wheel, so it is compiled in-place under emulation (slow, once).
python3 -m pip install --quiet --target="${SYSROOT}/usr/lib/python3/dist-packages" rich
chroot "${SYSROOT}" /bin/sh -c '
  export PIP_BREAK_SYSTEM_PACKAGES=1
  python3 -c "import ml_dtypes" 2>/dev/null && exit 0
  pip3 install -U --ignore-installed setuptools wheel pybind11
  pip3 install --no-build-isolation --no-deps "ml_dtypes>=0.5.4"
'

chroot "${SYSROOT}" /usr/bin/python3 -c "
import sys, numpy
print('rootfs ready:', sys.byteorder, 'endian | python', sys.version.split()[0],
      '| numpy', numpy.__version__)
"
