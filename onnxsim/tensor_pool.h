/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorPool: a ref-counted, name-keyed store of tensor byte buffers, with a
 * HuggingFace safetensors (https://github.com/huggingface/safetensors) file
 * as its serialization format.
 *
 * Motivation (reducing onnx::TensorProto data copies): onnx::TensorProto
 * physically owns its `raw_data` bytes as a plain std::string, so ordinary
 * protobuf copies (ModelProto assignment, onnx-optimizer's Import/Export
 * round trip, a Python trampoline hop) each deep-copy every initializer's
 * payload -- see onnxsim.cpp's OptimizeFixed move-through-the-round-trip fix
 * for issue #633, which works around exactly this for one specific call
 * site. TensorPool generalizes the same idea for the two places a real copy
 * is otherwise unavoidable:
 *
 *   1. Loading weights from disk: onnx's own external-data loader (and a
 *      naive safetensors reader) memcpy's each tensor's slice out of the
 *      file into a fresh TensorProto.raw_data. LoadSafetensors instead reads
 *      the file into ONE owned buffer and hands out std::shared_ptr-aliased
 *      views into it -- one copy total (the disk read), not one per tensor,
 *      and every Entry a caller Finds() shares that same buffer for free.
 *   2. Saving weights back out: SaveSafetensors writes each entry straight
 *      from its own buffer -- it never concatenates the pool into one
 *      in-memory blob first.
 *
 * TensorPool itself is dependency-free of onnx/protobuf (only the *integer*
 * ONNX dtype codes from tensor_pool_dtype.h), so it builds and unit-tests
 * standalone -- mirrors the dlpack_dtype.h / dlpack_bridge.h split. The
 * onnx::TensorProto <-> TensorPool glue lives in tensor_pool_bridge.h; see
 * that header's comment for how (and where) it's safe to plug a pool into
 * onnxsim's existing passes -- notably, several of them (onnxsim.cpp's
 * IsAllZeroTensor and integer-tensor extraction, dlpack_bridge.h's
 * FromTensorProtoBorrowing) already special-case and skip
 * onnx::TensorProto::EXTERNAL tensors, so a pool reference is only safe to
 * hand them once it has been hydrated back to an ordinary in-memory tensor.
 *
 * File format (https://github.com/huggingface/safetensors#format):
 *   [8 bytes] header length N, u64 little-endian
 *   [N bytes] UTF-8 JSON header: a flat object of
 *     {"<name>": {"dtype": "<code>", "shape": [...],
 *                 "data_offsets": [begin, end]}, ...},
 *     plus an optional "__metadata__" entry (arbitrary string->string map,
 *     skipped by this reader)
 *   [rest of file] raw tensor bytes, concatenated at the offsets the header
 *     describes, relative to the end of the header
 * This is onnx's own "raw external data in a file" layout with a JSON
 * manifest bolted on: a tensor is addressed the same way (an offset + length
 * range in a file), just keyed by name via the header instead of by
 * per-tensor location/offset/length entries living in the TensorProto
 * itself. That means a TensorPool's file is BOTH a standard safetensors file
 * (openable by the `safetensors` Python package, HF `transformers`/
 * `diffusers`, etc., with no onnxsim involved) AND usable, unmodified, as the
 * backing file for onnx's own classic external-data mechanism, by pointing
 * each TensorProto's external_data offset past the header (see
 * tensor_pool_bridge.h's ExportModelWithSafetensors).
 *
 * Byte order: like onnx::TensorProto::raw_data (which ONNX's spec fixes to
 * little-endian on every host) and every safetensors file actually produced
 * in the wild, an Entry's `data` is *always* little-endian bytes, regardless
 * of host byte order. TensorPool never interprets or swaps them -- only the
 * file's own 8-byte header-length prefix, which this pool reads/writes byte
 * by byte, needs (and gets) explicit little-endian handling for correctness
 * on a big-endian host (this repo tests on s390x; see dlpack_dtype.h's
 * kRawDataIsHostOrder for the *other* boundary, where such bytes are
 * eventually swapped into host order for arithmetic).
 */
#ifndef ONNXSIM_TENSOR_POOL_H_
#define ONNXSIM_TENSOR_POOL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace onnxsim {
namespace tensor_pool {

// A named tensor's data as a zero-copy view into whatever storage TensorPool
// keeps alive on its behalf. `owner` keeps `data` alive (it may be an
// aliasing shared_ptr into a much larger buffer, e.g. one Entry per tensor in
// a loaded safetensors file all alias the same underlying read); copying an
// Entry is a shared_ptr refcount bump, never a byte copy.
struct Entry {
  int32_t dtype = 0;  // an OnnxDtype from tensor_pool_dtype.h
  std::vector<int64_t> shape;
  std::shared_ptr<const char[]> owner;
  std::string_view data;  // dtype's raw little-endian bytes; aliases *owner

  size_t nbytes() const { return data.size(); }
};

class TensorPool {
 public:
  // Store `bytes` under `name`, taking ownership without copying (the string
  // is moved in and becomes the pool's sole owner of that allocation).
  // Overwrites any existing entry of the same name.
  void Add(const std::string& name, int32_t dtype, std::vector<int64_t> shape,
           std::string&& bytes);

  // Store bytes already owned elsewhere via a shared_ptr -- e.g. a view
  // produced by another pool's LoadSafetensors, or a sub-range of an entry
  // this pool already holds. No copy either way.
  void Add(const std::string& name, int32_t dtype, std::vector<int64_t> shape,
           std::shared_ptr<const char[]> owner, std::string_view data);

  const Entry* Find(const std::string& name) const;
  bool Erase(const std::string& name);
  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }

  // Write every entry to a .safetensors file at `path`, in the pool's
  // iteration (name-sorted) order. Each tensor is written straight from its
  // own Entry::data -- never concatenated into one in-memory blob first --
  // so saving performs no tensor-data copies of its own beyond the OS's
  // ordinary write() buffering. Throws std::runtime_error if any pooled
  // dtype has no safetensors representation (see tensor_pool_dtype.h), or on
  // I/O failure.
  //
  // When `data_offsets_out` is non-null, it is filled with each written
  // tensor's [begin, end) byte range *relative to the end of the header*
  // (exactly the file's own "data_offsets" header field) -- callers that
  // need the tensor's *absolute* file offset (e.g. to fill in a
  // TensorProto's external_data, as tensor_pool_bridge.h's
  // ExportModelWithSafetensors does) add HeaderPrefixSize(path)'s result.
  void SaveSafetensors(const std::string& path,
                       std::map<std::string, std::pair<uint64_t, uint64_t>>*
                           data_offsets_out = nullptr) const;

  // Replace this pool's contents with every tensor described by the
  // .safetensors file at `path`. Reads the whole file into one owned buffer
  // and gives every Entry a std::shared_ptr that aliases *that one buffer*
  // (via shared_ptr's aliasing constructor) -- one copy total (the disk
  // read), not one per tensor. Throws std::runtime_error on I/O failure, a
  // malformed header, or a tensor whose data_offsets fall outside the file.
  void LoadSafetensors(const std::string& path);

 private:
  std::map<std::string, Entry> entries_;
};

// Bytes of `path`'s safetensors preamble (the 8-byte length prefix plus the
// JSON header itself) -- i.e. where the raw tensor data segment begins, and
// so what SaveSafetensors's per-tensor data_offsets are relative to. Reads
// only the 8-byte length prefix, not the whole file. Throws
// std::runtime_error on I/O failure.
uint64_t HeaderPrefixSize(const std::string& path);

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_H_
