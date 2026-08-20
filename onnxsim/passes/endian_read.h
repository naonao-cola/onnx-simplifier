// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Endian-safe reads of a Tensor's raw_data, for every onnxsim pass that reads
// a constant's numeric values directly -- e.g. to quantize a weight, or to
// match a fusion pattern's literal scalar operands (0.5, sqrt(2), ...).
//
// ONNX fixes the wire byte order of TensorProto::raw_data to little-endian on
// every host (see onnxsim::dlpack::kRawDataIsHostOrder's doc comment,
// dlpack_dtype.h, for the identical concern at the DLPack constant-folding
// boundary). onnx-optimizer's in-memory Tensor (onnx/common/ir.h), used
// throughout onnxsim's own passes, copies that wire byte range into its own
// raw_data storage completely unchanged (see ir_pb_converter.cc's
// importTensor) -- so on a little-endian host Tensor::data<T>() already
// returns host-order values, but on a big-endian host it returns the wire
// bytes verbatim, misread as if they were already native-order. A typed
// field (Tensor::floats(), ::doubles(), ...) needs no such swap either way:
// protobuf decodes those into host-order scalars itself, which is why every
// read site here only ever swaps the is_raw_data() branch.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "dlpack_dtype.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Copies `numel` elements starting at `raw` (a Tensor::data<T>() pointer into
// raw_data storage) into a host-order vector<T>, swapping each element's
// bytes only on a big-endian host -- a plain memcpy on a little-endian one,
// where raw_data already agrees with the host.
template <typename T>
inline std::vector<T> ReadRawDataHostOrder(const T* raw, int64_t numel) {
  std::vector<T> out(static_cast<size_t>(numel));
  std::memcpy(out.data(), raw, out.size() * sizeof(T));
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                      out.size() * sizeof(T), sizeof(T));
  }
  return out;
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
