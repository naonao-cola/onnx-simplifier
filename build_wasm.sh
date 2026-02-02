#!/usr/bin/env bash
set -ex

# Check if emcmake is available
command -v emcmake

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WITH_NODE_RAW_FS=${1:-OFF}

cd $SCRIPT_DIR
if which protoc ; then
    PROTOC=$(which protoc)
else
    . ./third_party/onnx-optimizer/third_party/onnx/workflow_scripts/protobuf/build_protobuf_unix.sh $(nproc) $PWD/protobuf
    PROTOC=$(which protoc)
fi

set -u -o pipefail

ORT_VER=1.23.2
if [ ! -d "third_party/onnxruntime-${ORT_VER}" ] ; then
    pushd third_party
    wget -q "https://github.com/microsoft/onnxruntime/archive/refs/tags/v${ORT_VER}.zip"
    unzip -q "v${ORT_VER}.zip"
    curl -L https://github.com/onnx/onnx/pull/7609.diff >> "onnxruntime-${ORT_VER}/cmake/patches/onnx/onnx.patch"
    popd
fi

mkdir -p build-wasm-node-$WITH_NODE_RAW_FS
cd build-wasm-node-$WITH_NODE_RAW_FS
emcmake cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-sNO_DISABLE_EXCEPTION_CATCHING" ..
cmake --build . -t onnxsim_bin
