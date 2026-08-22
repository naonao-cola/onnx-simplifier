#include "onnxsim.h"

#include <google/protobuf/arena.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ONNXSIM_HAS_ORT
// Both ways CMake obtains ONNX Runtime (an official release, or
// cmake/build_ort.cmake's own from-source ExternalProject build) now install
// their public headers flat (e.g. onnxruntime_cxx_api.h directly under the
// include root) -- see ONNXSIM_ORT_FLAT_HEADERS in CMakeLists.txt /
// cmake/build_ort.cmake. Byte order is handled with C++20 std::endian in
// dlpack_dtype.h, so this does not depend on ORT's internal
// core/common/endian.h (which neither ships).
#include "onnxruntime_cxx_api.h"
#endif
#include "constant_folding.h"
#include "contrib_schemas.h"
#include "custom_optimizer_passes.h"
#include "dlpack_bridge.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/graph_shape_inference.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/ir_pb_converter_internal.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/logging.h"
#include "partial_shape_eval.h"
#include "passes/quantize_bf16.h"
#include "passes/quantize_conv_common.h"
#include "passes/quantize_fp16.h"
#include "passes/quantize_fp8.h"
#include "passes/quantize_matmul_common.h"
#include "passes/static_quantize_matmul.h"
#include "profiler.h"
#include "sym_shape_infer.h"
#include "sym_value_eval.h"

// A 128-bit fingerprint of a model, used by FixedPointFn to detect when an
// iteration stopped changing the model without keeping a second full ModelProto
// around just for the comparison. Two models with the same fingerprint are
// treated as equal; the odds of a false match are ~2^-128 per comparison, and a
// false match would only stop simplification one round early (the model stays
// valid), never produce an incorrect model.
struct ModelFingerprint {
  uint64_t h1;
  uint64_t h2;
  bool operator==(const ModelFingerprint& other) const {
    return h1 == other.h1 && h2 == other.h2;
  }
};

// Mixes ``n`` bytes at ``data`` into the two rolling hash accumulators (FNV-1a
// and a splitmix-style mix), 8 bytes (one machine word) at a time rather than
// one byte at a time -- cutting the number of loop iterations, and with them
// most of the loop/branch overhead, by roughly 8x versus a byte-at-a-time
// mix, for identical byte coverage and order.
void MixBytes(const char* data, size_t n, uint64_t& h1, uint64_t& h2) {
  const size_t n_words = n / sizeof(uint64_t);
  for (size_t i = 0; i < n_words; ++i) {
    // memcpy, not a reinterpret_cast dereference: ``data`` is not guaranteed
    // 8-byte aligned (it may be a chunk of a streaming buffer, or a
    // std::string's internal buffer), and an unaligned uint64_t load is
    // undefined behavior in C++. Any decent compiler lowers this fixed-size
    // memcpy to a single unaligned load.
    uint64_t word;
    std::memcpy(&word, data + i * sizeof(uint64_t), sizeof(uint64_t));
    h1 = (h1 ^ word) * 1099511628211ULL;
    h2 = (h2 + word) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
  // Fewer than 8 trailing bytes: fall back to a byte-at-a-time mix so every
  // byte still participates in the hash.
  for (size_t i = n_words * sizeof(uint64_t); i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(data[i]);
    h1 = (h1 ^ c) * 1099511628211ULL;
    h2 = (h2 + c) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
}

// A protobuf output stream that hashes bytes as they are handed out by the
// serializer instead of collecting them into a buffer. Fingerprint() used to
// call ``model.SerializeAsString()`` -- which allocates a buffer the size of
// the whole serialized model (hundreds of MB of initializer weight data on a
// large model) and serializes into it -- and then made a second, separate
// pass over that buffer to hash it. Handing the serializer a small, reused
// scratch buffer instead avoids that large allocation (and the deallocation
// moments later): each time the serializer fills it, this stream
// mixes it into the running hash and hands the same buffer back out, so the
// whole model is hashed in bounded extra memory regardless of its size.
class HashingOutputStream final
    : public google::protobuf::io::ZeroCopyOutputStream {
 public:
  bool Next(void** data, int* size) override {
    FlushPending();
    *data = buf_.data();
    *size = static_cast<int>(buf_.size());
    pending_ = static_cast<int>(buf_.size());
    return true;
  }
  void BackUp(int count) override { pending_ -= count; }
  int64_t ByteCount() const override { return byte_count_; }

  // Call after serialization completes to mix in the final partial buffer.
  void Finish() { FlushPending(); }

  uint64_t h1() const { return h1_; }
  uint64_t h2() const { return h2_; }
  size_t total_bytes() const { return static_cast<size_t>(byte_count_); }

 private:
  void FlushPending() {
    if (pending_ <= 0) {
      return;
    }
    MixBytes(buf_.data(), static_cast<size_t>(pending_), h1_, h2_);
    byte_count_ += pending_;
    pending_ = 0;
  }

  // 64 KiB: large enough that the per-Next() call overhead is negligible next
  // to the memory-bandwidth-bound hashing work, small enough to stay resident
  // in L1/L2 cache across the mix. Heap-allocated, not a fixed-size array
  // member: this object lives on the call stack of Fingerprint(), which is
  // itself invoked from deep inside the (possibly nested) fixed-point
  // machinery, and the WASM build's stack is only tens of KiB total -- a
  // 64 KiB stack-resident array here reliably overflowed it (issue caught by
  // the WASM build's own smoke test, "Aborted(stack overflow ...)" on the
  // very first Simplify() call). std::vector's control block is a few
  // pointer-sized words on the stack; the buffer itself is heap-backed.
  std::vector<char> buf_ = std::vector<char>(1 << 16);
  int pending_ = 0;
  int64_t byte_count_ = 0;
  uint64_t h1_ = 1469598103934665603ULL;  // FNV-1a offset basis
  uint64_t h2_ = 0;
};

ModelFingerprint Fingerprint(const onnx::ModelProto& model) {
  // ModelProto contains no protobuf ``map<>`` fields, so serialization order is
  // stable and equal models serialize to identical bytes.
  //
  // Fingerprint() runs at every step of every (possibly nested) fixed point,
  // so on a model with hundreds of MB of initializer weight data -- which
  // dominates the serialized size but is untouched by most rounds (shape
  // inference, the onnx-optimizer passes) -- it still has to read all of that
  // data, every round, just to prove nothing changed. Profiling a 265 MB
  // model showed this was a multi-second fraction (over half of total wall
  // time on that model) of total simplification time, so it is worth
  // streaming through a small buffer (see HashingOutputStream) rather than
  // materializing the whole serialized model first.
  HashingOutputStream stream;
  // SerializeToZeroCopyStream only returns false if a Next() call on the
  // stream fails (out of disk space, a broken pipe, ...); HashingOutputStream
  // writes to memory and its Next() always succeeds, so this cannot fail.
  model.SerializeToZeroCopyStream(&stream);
  stream.Finish();
  uint64_t h1 = stream.h1();
  uint64_t h2 = stream.h2();
  h2 ^= stream.total_bytes();
  return {h1, h2};
}

// Alternately apply ``f1`` and ``f2`` until the model stops changing (a joint
// fixed point) or ``max_iters`` alternations elapse. Each application produces
// a fresh model, so ``model`` is move-assigned in place and only a single
// ModelProto is held live across the loop; convergence is detected by comparing
// the fingerprints of consecutive states rather than keeping the previous
// ModelProto for a ``MessageDifferencer::Equals`` call. This mirrors the
// original consecutive-pair comparison exactly -- it stops as soon as the last
// applied function left the model unchanged -- while roughly halving the number
// of full model copies held at once (which matters because these fixed points
// nest).
// The transforms mutate the model in place (``std::function<void(T&)>``), so a
// transform that already works in place (e.g. ``_InferShapes``) makes no copy
// at all, and one that must build a fresh model (e.g. ``Optimize``, whose
// underlying ``OptimizeFixed`` returns a new proto) move-assigns it back. The
// returned function likewise mutates in place, so it composes when these fixed
// points nest and a single ModelProto is threaded through the whole thing.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    // Profiled separately from the transforms it gates: on large models this
    // convergence check is not free (see Fingerprint()'s comment), so its cost
    // should be visible in its own right rather than silently inflating
    // whichever of f1/f2's spans happens to run next. A no-op unless
    // ONNXSIM_PROFILE is set.
    auto fingerprint = [](const onnx::ModelProto& m) {
      onnxsim::ProfiledScope scope("Fingerprint");
      return Fingerprint(m);
    };
    size_t _max_iters = max_iters;
    f1(model);
    ModelFingerprint fp_prev = fingerprint(model);
    f2(model);
    ModelFingerprint fp_cur = fingerprint(model);
    while (_max_iters-- > 0) {
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f1(model);
      fp_prev = fp_cur;
      fp_cur = fingerprint(model);
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f2(model);
      fp_prev = fp_cur;
      fp_cur = fingerprint(model);
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// Same convergence algorithm as the ``Fingerprint``-based ``FixedPointFn``
// above, specialized for transforms that can cheaply report whether they
// changed the model themselves (``f1``/``f2`` return true iff they did),
// instead of hashing the whole serialized model after every call. This skips
// ``Fingerprint()`` entirely, which matters most here because this is the
// innermost, most-frequently-run fixed point (``OptAndShape`` below runs
// every simplification round). It is only as safe as ``f1``/``f2``'s own
// signal: both onnx-optimizer's per-pass transform counts and onnx's
// InferShapes value-change count are exact for onnxsim's pass list (no
// pass with an ``Empty``/uncounted analysis type is used), so a ``false``
// return means that call provably made no change -- not just "probably".
//
// Convergence requires a whole *round* (one ``f1`` call followed by one
// ``f2`` call) to report no change from either, not just the most recent
// call: ``f1`` reporting false only means it found nothing new, not that
// ``f2``'s previous call is done making downstream cleanup available (e.g.
// a Complete-efficiency onnx-optimizer pass firing late in
// ``OptimizeGraphFixed``'s own pass-list order can leave a now-dead node
// that an earlier-ordered pass in that same call never got a chance to
// revisit -- ``f1`` correctly reports no new shape info either way, but
// ``f2`` still has cleanup work left for its *next* call). Stopping right
// after a lone ``f1`` false, as an earlier version of this function did,
// silently left that cleanup undone whenever nothing above this loop
// happened to re-drive it; matching a whole round instead of a single call
// closes that gap unconditionally, with no dependence on some caller
// incidentally running this again.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<bool(T&)>& f1,
                                     const std::function<bool(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    size_t _max_iters = max_iters;
    while (_max_iters-- > 0) {
      const bool c1 = f1(model);
      const bool c2 = f2(model);
      if (!c1 && !c2) {
        if (converged) {
          *converged = true;
        }
        return;
      }
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<bool(T&)>& f1,
                                     const std::function<bool(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// A no-op in-place transform (mutates nothing), used when shape inference or
// constant folding is disabled.
void Identity(onnx::ModelProto&) {}

// Recursively collect the op types of operators that live in ONNX's *default*
// domain but have no registered schema. These are custom operators -- most
// commonly TensorRT plugins such as ``BatchedNMS_TRT`` or ``EfficientNMS_TRT``
// -- that were exported into the default domain instead of a vendor-specific
// one.
//
// Custom ops that already live in a non-default domain (e.g. ``com.microsoft``
// or ``TRT``) are intentionally ignored: onnx::checker::check_model already
// tolerates unknown ops in non-standard domains, which is exactly the manual
// workaround reported in GitHub issue #220.
void CollectCustomDefaultDomainOps(const onnx::GraphProto& graph,
                                   int default_opset_version,
                                   std::set<std::string>& custom_ops) {
  for (const auto& node : graph.node()) {
    const std::string& domain = node.domain();
    const bool is_default_domain = domain.empty() || domain == "ai.onnx";
    if (is_default_domain &&
        onnx::OpSchemaRegistry::Schema(node.op_type(), default_opset_version,
                                       /*domain=*/"") == nullptr) {
      custom_ops.insert(node.op_type());
    }
    // Recurse into subgraphs held in node attributes (If/Loop/Scan bodies).
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectCustomDefaultDomainOps(attr.g(), default_opset_version,
                                      custom_ops);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectCustomDefaultDomainOps(subgraph, default_opset_version,
                                      custom_ops);
      }
    }
  }
}

// Register a permissive placeholder schema for every default-domain custom op
// found in ``model``. Without a schema, onnx::checker::check_model rejects the
// model with "No Op registered for <op> with domain_version of <n>" and
// simplification never even starts (GitHub issues #107 and #220). The
// placeholder accepts any number of inputs/outputs of any tensor type and any
// attributes, so the checker passes and the op is preserved untouched through
// simplification. It carries no shape/type inference function, so shape
// inference simply flows past the op as before.
void RegisterCustomDefaultDomainOpSchemas(const onnx::ModelProto& model) {
  int default_opset_version = 1;
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      default_opset_version =
          std::max(default_opset_version, static_cast<int>(opset.version()));
    }
  }

  std::set<std::string> custom_ops;
  CollectCustomDefaultDomainOps(model.graph(), default_opset_version,
                                custom_ops);

  for (const auto& op_type : custom_ops) {
    onnx::OpSchema schema;
    schema.SetName(op_type)
        .SetDomain("")
        .SinceVersion(1)
        .SetDoc(
            "Placeholder schema registered by onnxsim for a custom operator "
            "(e.g. a TensorRT plugin) exported into the default ONNX domain, "
            "so "
            "that the model passes validation and is simplified with the "
            "operator preserved unchanged.")
        .Input(0, "inputs", "Variadic inputs of the custom operator.", "T",
               onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
               /*min_arity=*/0)
        .Output(0, "outputs", "Variadic outputs of the custom operator.", "T",
                onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
                /*min_arity=*/0)
        .TypeConstraint("T", onnx::OpSchema::all_tensor_types(),
                        "Allow inputs and outputs of any tensor type.")
        // Custom ops carry arbitrary, plugin-specific attributes; accept them
        // all rather than trying to enumerate them.
        .AllowUncheckedAttributes();
    // Never fail or throw: a duplicate registration (e.g. simplifying two
    // models that use the same custom op in one process) is a harmless no-op.
    onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                         /*fail_duplicate_schema=*/false,
                         /*fail_with_exception=*/false);
  }
}

// Collect the names of every node in `graph`, descending into subgraphs held in
// node attributes (If/Loop/Scan bodies). Used to de-duplicate the names later
// assigned to nameless nodes.
void CollectNodeNames(const onnx::GraphProto& graph,
                      std::set<std::string>& names) {
  for (const auto& node : graph.node()) {
    if (!node.name().empty()) {
      names.insert(node.name());
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectNodeNames(attr.g(), names);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectNodeNames(subgraph, names);
      }
    }
  }
}

// Give a unique, deterministic name to every node that has none. Nodes without
// a name survive simplification unnamed -- either because they were nameless in
// the input model or because an onnx-optimizer pass created a replacement node
// without setting a name -- which trips up downstream tooling that keys on node
// names (issue #269). Each generated name is derived from the op type plus a
// running counter and de-duplicated against every name already present in the
// graph (including names generated earlier in this pass). Subgraphs are handled
// recursively so nodes inside If/Loop/Scan bodies are named too.
void AssignMissingNodeNames(onnx::GraphProto& graph,
                            std::set<std::string>& used_names,
                            size_t& counter) {
  for (auto& node : *graph.mutable_node()) {
    if (node.name().empty()) {
      std::string name;
      do {
        name = node.op_type() + "_" + std::to_string(counter++);
      } while (used_names.count(name) > 0);
      used_names.insert(name);
      node.set_name(name);
    }
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        AssignMissingNodeNames(*attr.mutable_g(), used_names, counter);
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        AssignMissingNodeNames(subgraph, used_names, counter);
      }
    }
  }
}

// Assign names to any nodes left nameless after simplification (issue #269).
void AssignMissingNodeNames(onnx::ModelProto& model) {
  std::set<std::string> used_names;
  CollectNodeNames(model.graph(), used_names);
  size_t counter = 0;
  AssignMissingNodeNames(*model.mutable_graph(), used_names, counter);
}

// Shape inference can leave behind a value_info entry that records a shape
// but never resolved an element type (e.g. observed on a real-world model's
// Reshape outputs inside attention blocks: graph_shape_inference.h's
// InferShapesOnGraph propagated the target shape but left elem_type at its
// default UNDEFINED). value_info is purely optional annotation -- nothing
// reads it as ground truth -- but an entry with elem_type == UNDEFINED is a
// malformed TypeProto, and onnx::checker::check_model does not reject it,
// while onnxruntime's own model loader is stricter and refuses to load the
// model at all with "Invalid tensor data type 0". Drop such entries instead
// of leaving broken metadata in the output model; the same op's actual
// output tensor is unaffected either way.
// Recurses into subgraphs (If/Loop/Scan bodies) for the same reason
// AssignMissingNodeNames does.
void DropIncompleteValueInfo(onnx::GraphProto& graph) {
  auto& value_info = *graph.mutable_value_info();
  value_info.erase(
      std::remove_if(value_info.begin(), value_info.end(),
                     [](const onnx::ValueInfoProto& vi) {
                       return vi.type().has_tensor_type() &&
                              vi.type().tensor_type().elem_type() ==
                                  onnx::TensorProto::UNDEFINED;
                     }),
      value_info.end());
  for (auto& node : *graph.mutable_node()) {
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        DropIncompleteValueInfo(*attr.mutable_g());
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        DropIncompleteValueInfo(subgraph);
      }
    }
  }
}

void DropIncompleteValueInfo(onnx::ModelProto& model) {
  DropIncompleteValueInfo(*model.mutable_graph());
}

void Check(const onnx::ModelProto& model) { onnx::checker::check_model(model); }

// Return the opset version the model imports for the default ONNX domain
// (represented as either the empty string or "ai.onnx"), or std::nullopt when
// the model does not import the default domain at all (a model made purely of
// custom-domain operators).
std::optional<int> DefaultOpsetVersion(const onnx::ModelProto& model) {
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return static_cast<int>(opset.version());
    }
  }
  return std::nullopt;
}

// Mirrors onnx_simplifier.py's remove_initializer_from_input: an initializer
// that also appears in graph.input is treated as a runtime input by
// onnxoptimizer's is_constant_initializer, which blocks value-baking fusions
// (fuse_bn_into_conv, ...) that only work on non-input initializers -- e.g.
// the plain Conv+BN chains of the opset-8 resnet101-v1-7 were left completely
// unsimplified without this. Removing it from the input list lets it fold
// like any other constant. Kept in sync with the Python function; see that
// one's own comment for the opset-6 floor (Cast's `to` attribute's INT
// encoding wasn't valid before opset 6, so bumping an old-IR/old-opset model
// to IR 4 here would let a later fusion emit a node the opset rejects).
void RemoveInitializerFromInput(onnx::ModelProto& model) {
  constexpr int kMinOpsetForInitializerFold = 6;
  if (model.ir_version() < 4) {
    const auto opset = DefaultOpsetVersion(model);
    if (!opset || *opset < kMinOpsetForInitializerFold) {
      return;
    }
  }
  std::set<std::string> initializer_names;
  for (const auto& init : model.graph().initializer()) {
    initializer_names.insert(init.name());
  }
  auto* inputs = model.mutable_graph()->mutable_input();
  bool removed_any = false;
  for (int i = inputs->size() - 1; i >= 0; --i) {
    if (initializer_names.count((*inputs)[i].name()) > 0) {
      inputs->erase(inputs->begin() + i);
      removed_any = true;
    }
  }
  if (removed_any && model.ir_version() < 4) {
    model.set_ir_version(4);
  }
}

// ONNX TensorProto element types onnxoptimizer's tensor-value hashing
// (cse_util.h) knows how to hash. Any other type makes
// eliminate_common_subexpression/eliminate_duplicate_initializer raise
// "no supported data type: <N>". Enumerates the *supported* types (rather
// than the unsupported ones) so an element type added to ONNX in the future
// is treated as unhashable by default instead of silently crashing. Mirrors
// onnx_simplifier.py's _CSE_HASHABLE_ELEM_TYPES.
bool IsCSEHashableElemType(int32_t elem_type) {
  switch (elem_type) {
    case onnx::TensorProto::UNDEFINED:
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT64:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::DOUBLE:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::COMPLEX128:
    case onnx::TensorProto::STRING:
      return true;
    default:
      return false;
  }
}

// Mirrors onnx_simplifier.py's _has_cse_unhashable_tensor /
// _iter_tensor_data_types: walks every tensor CSE might hash -- initializers,
// t/ts node attributes, recursing into subgraphs -- looking for an element
// type IsCSEHashableElemType rejects.
bool GraphHasCSEUnhashableTensor(const onnx::GraphProto& graph) {
  for (const auto& init : graph.initializer()) {
    if (!IsCSEHashableElemType(init.data_type())) {
      return true;
    }
  }
  for (const auto& node : graph.node()) {
    for (const auto& attr : node.attribute()) {
      if (attr.has_t() && !IsCSEHashableElemType(attr.t().data_type())) {
        return true;
      }
      for (const auto& t : attr.tensors()) {
        if (!IsCSEHashableElemType(t.data_type())) {
          return true;
        }
      }
      if (attr.has_g() && GraphHasCSEUnhashableTensor(attr.g())) {
        return true;
      }
      for (const auto& g : attr.graphs()) {
        if (GraphHasCSEUnhashableTensor(g)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Mirrors onnx_simplifier.py's overwrite_input_shapes loop: for each named
// graph input, overwrite its shape's dims with the given values, skipping
// any non-positive entry (which means "keep the original, possibly dynamic,
// dimension" rather than hardcoding an invalid size such as 0 -- GitHub
// issue #237). Throws std::runtime_error if a name isn't a graph input
// (matching onnx_simplifier.py's RuntimeError for the same case).
void ApplyInputShapeOverwrite(
    onnx::ModelProto& model,
    const std::unordered_map<std::string, std::vector<int64_t>>&
        overwrite_input_shapes) {
  for (const auto& [name, shape] : overwrite_input_shapes) {
    bool found = false;
    for (auto& input : *model.mutable_graph()->mutable_input()) {
      if (input.name() != name) {
        continue;
      }
      found = true;
      auto* dims = input.mutable_type()
                       ->mutable_tensor_type()
                       ->mutable_shape()
                       ->mutable_dim();
      for (int i = 0; i < dims->size() && i < static_cast<int>(shape.size());
           ++i) {
        if (shape[i] > 0) {
          dims->Mutable(i)->set_dim_value(shape[i]);
        }
      }
    }
    if (!found) {
      throw std::runtime_error("The model doesn't have input named \"" + name +
                               "\"");
    }
  }
}

// Mirrors onnx_simplifier.py's remove_unused_output: drops the named graph
// outputs. Downstream dead-end elimination cleans up nodes that only fed
// them. Throws std::runtime_error if a name isn't a graph output (matching
// onnx_simplifier.py's RuntimeError for the same case).
void RemoveUnusedOutputs(onnx::ModelProto& model,
                         const std::vector<std::string>& unused_output) {
  for (const auto& name : unused_output) {
    bool found = false;
    for (const auto& output : model.graph().output()) {
      if (output.name() == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("The model doesn't have output named \"" + name +
                               "\"");
    }
  }
  auto* outputs = model.mutable_graph()->mutable_output();
  google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> kept;
  for (auto& output : *outputs) {
    if (std::find(unused_output.begin(), unused_output.end(), output.name()) ==
        unused_output.end()) {
      *kept.Add() = std::move(output);
    }
  }
  outputs->Swap(&kept);
}

// Convert the default ONNX domain of the model to target_version using onnx's
// own version converter. Only the default ONNX domain opset is changed;
// custom/other-domain opset imports are left untouched. Returns the model
// unchanged when it is already at the requested version or does not import the
// default domain.
onnx::ModelProto ConvertOpsetVersion(const onnx::ModelProto& model,
                                     int target_version) {
  const auto current = DefaultOpsetVersion(model);
  if (!current || *current == target_version) {
    return model;
  }
  return onnx::version_conversion::ConvertVersion(model, target_version);
}

// Shared schema setup for the single-pass debug helpers below, mirroring the
// head of ``Simplify``: teach shape inference about ONNX Runtime's quantized
// contrib ops and register permissive placeholders for custom ops exported into
// the default ONNX domain, so neither shape inference nor a later checker
// rejects the model. Unlike ``Simplify`` these helpers do not ``Check`` the
// model up front -- the point of running a step in isolation is to debug a
// model that may not yet be fully valid.
void PrepareSchemasForDebug(const onnx::ModelProto& model) {
  onnxsim::RegisterContribOpSchemas();
  FixupSchemaDeterminism();
  RegisterCustomDefaultDomainOpSchemas(model);
}

onnx::ModelProto InferShapesOnce(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnx::ModelProto out = model;
  _InferShapes(out);
  return out;
}

onnx::ModelProto PropagateDataOnce(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnx::ModelProto out = model;
  _EvalPartialShape(out);
  return out;
}

onnx::ModelProto FoldConstantOnce(const ModelExecutor& executor,
                                  const onnx::ModelProto& model,
                                  size_t tensor_size_threshold,
                                  bool initializers_as_constants) {
  PrepareSchemasForDebug(model);
  // ``_FoldConstant`` reads these globals (tensor-size cap and the
  // initializers-as-constants policy), just as ``Simplify`` sets them.
  config.tensor_size_threshold = tensor_size_threshold;
  config.initializers_as_constants = initializers_as_constants;
  config.optimizer_passes.clear();
  onnx::ModelProto out = model;
  // Mirror ``Simplify``'s FoldConstant step: partial shape evaluation first
  // turns Shape/Gather-on-shape into constants that the ordinary constant
  // folder can then propagate.
  _EvalPartialShape(out);
  out = _FoldConstant(executor, std::move(out));
  return out;
}

onnx::ModelProto QuantizeDynamic(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_matmul (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"dynamic_quantize_matmul"});
}

onnx::ModelProto QuantizeTernary(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_ternary_matmul (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"dynamic_quantize_ternary_matmul"});
}

onnx::ModelProto QuantizeWeightOnly(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_matmul",
                                      "weight_only_quantize_conv"});
}

onnx::ModelProto QuantizeWeightOnlyInt4(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_int4_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_int4_matmul",
                                      "weight_only_quantize_int4_conv"});
}

std::vector<std::string> ListQuantizableActivations(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::MatMulLikeInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchMatMulLike(node, info)) {
      continue;
    }
    if (info.x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    const onnx::Tensor* w_t = onnx::optimization::FetchConstantTensor(info.w);
    if (w_t == nullptr ||
        w_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      continue;
    }
    if (seen.insert(info.x->uniqueName()).second) {
      names.push_back(info.x->uniqueName());
    }
  }
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::ConvInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchConv(node, info)) {
      continue;
    }
    if (info.x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    const onnx::Tensor* w_t = onnx::optimization::FetchConstantTensor(info.w);
    if (w_t == nullptr ||
        w_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 3) {
      continue;
    }
    if (seen.insert(info.x->uniqueName()).second) {
      names.push_back(info.x->uniqueName());
    }
  }
  return names;
}

onnx::ModelProto QuantizeStatic(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers static_quantize_matmul (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // static_quantize_matmul reads this global, the same way onnxsim's other
  // passes read `config` -- see StaticQuantizationCalibrationRanges's doc
  // comment.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"static_quantize_matmul",
                                      "static_quantize_conv"});
}

std::vector<std::string> ListQOperatorQuantizableOutputs(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::MatMulLikeInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchMatMulLike(node, info)) {
      continue;
    }
    if (info.x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    const onnx::Tensor* w_t = onnx::optimization::FetchConstantTensor(info.w);
    if (w_t == nullptr ||
        w_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      continue;
    }
    if (seen.insert(node->output()->uniqueName()).second) {
      names.push_back(node->output()->uniqueName());
    }
  }
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::ConvInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchConv(node, info)) {
      continue;
    }
    if (info.x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    const onnx::Tensor* w_t = onnx::optimization::FetchConstantTensor(info.w);
    if (w_t == nullptr ||
        w_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 3) {
      continue;
    }
    if (info.bias != nullptr) {
      // qoperator_quantize_conv only rewrites a Conv whose bias (if any) is a
      // constant float32 [Cout] tensor it can pre-quantize to INT32 -- see
      // that pass's doc comment. Skip listing this Conv's output otherwise,
      // since it will never actually be rewritten.
      const onnx::Tensor* b_t =
          onnx::optimization::FetchConstantTensor(info.bias);
      if (b_t == nullptr ||
          b_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
          b_t->sizes().size() != 1) {
        continue;
      }
    }
    if (seen.insert(node->output()->uniqueName()).second) {
      names.push_back(node->output()->uniqueName());
    }
  }
  return names;
}

onnx::ModelProto QuantizeQOperator(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_matmul/_conv read the same calibration-ranges global
  // static_quantize_matmul/_conv do (see StaticQuantizationCalibrationRanges's
  // doc comment) -- both are keyed by tensor name, and QOperator format's
  // ranges are a superset (activation names plus output names) of QDQ
  // format's, so sharing the map is safe as long as only one of
  // QuantizeStatic/QuantizeQOperator runs per call, which OptimizeFixed's
  // single pass-name list here ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_matmul",
                                      "qoperator_quantize_conv"});
}

onnx::ModelProto QuantizeFp16(const onnx::ModelProto& model,
                              bool keep_io_types) {
  PrepareSchemasForDebug(model);
  // Registers quantize_fp16 (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quantize_fp16 reads this the same way static_quantize_matmul.h's passes
  // read StaticQuantizationCalibrationRanges() -- OptimizeFixed's pass-name
  // list has no way to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuantizeFp16KeepIoTypes() = keep_io_types;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"quantize_fp16"});
}

onnx::ModelProto QuantizeBf16(const onnx::ModelProto& model,
                              bool keep_io_types) {
  PrepareSchemasForDebug(model);
  // Registers quantize_bf16 (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quantize_bf16 reads this the same way quantize_fp16 reads
  // QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no way
  // to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuantizeBf16KeepIoTypes() = keep_io_types;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"quantize_bf16"});
}

onnx::ModelProto QuantizeFp8(const onnx::ModelProto& model,
                             const std::string& format, bool keep_io_types) {
  onnx::optimization::onnxsim_passes::Float8Format target_format;
  if (format == "e4m3") {
    target_format = onnx::optimization::onnxsim_passes::Float8Format::kE4M3FN;
  } else if (format == "e5m2") {
    target_format = onnx::optimization::onnxsim_passes::Float8Format::kE5M2;
  } else {
    throw std::invalid_argument(
        "QuantizeFp8: format must be \"e4m3\" or \"e5m2\", got \"" + format +
        "\"");
  }
  PrepareSchemasForDebug(model);
  // Registers quantize_fp8 (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quantize_fp8 reads these the same way quantize_fp16 reads
  // QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no way
  // to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuantizeFp8TargetFormat() = target_format;
  onnx::optimization::onnxsim_passes::QuantizeFp8KeepIoTypes() = keep_io_types;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"quantize_fp8"});
}

// Records the options this Simplify() call actually used (skip_optimizers
// reflects any auto-added unhashable-tensor protections) as string
// key/value pairs in the model's metadata_props, namespaced under
// "onnxsim:" so they cannot collide with the model's own metadata. Lets a
// downstream consumer see how a model was simplified without having to have
// kept the original call around. Replaces any pre-existing "onnxsim:*"
// entries (e.g. left by a previous simplify() call) rather than
// accumulating duplicates across repeated simplification.
void RecordSimplifyOptionsMetadata(
    onnx::ModelProto& model,
    const std::optional<std::vector<std::string>>& skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, bool initializers_as_constants,
    bool include_inline_functions, bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output) {
  auto join = [](const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) {
        out += ",";
      }
      out += v[i];
    }
    return out;
  };

  std::vector<std::pair<std::string, std::string>> options;
  options.emplace_back("onnxsim:skip_optimizers",
                       skip_optimizers ? join(*skip_optimizers) : "<all>");
  options.emplace_back("onnxsim:constant_folding",
                       constant_folding ? "true" : "false");
  options.emplace_back("onnxsim:shape_inference",
                       shape_inference ? "true" : "false");
  options.emplace_back("onnxsim:tensor_size_threshold",
                       std::to_string(tensor_size_threshold));
  options.emplace_back(
      "onnxsim:target_opset_version",
      target_opset_version ? std::to_string(*target_opset_version) : "");
  options.emplace_back("onnxsim:initializers_as_constants",
                       initializers_as_constants ? "true" : "false");
  options.emplace_back("onnxsim:include_inline_functions",
                       include_inline_functions ? "true" : "false");
  options.emplace_back("onnxsim:mutable_initializer",
                       mutable_initializer ? "true" : "false");
  if (overwrite_input_shapes) {
    std::string s;
    bool first = true;
    for (const auto& [name, shape] : *overwrite_input_shapes) {
      if (!first) {
        s += ";";
      }
      first = false;
      s += name + ":";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
          s += ",";
        }
        s += std::to_string(shape[i]);
      }
    }
    options.emplace_back("onnxsim:overwrite_input_shapes", s);
  }
  if (unused_output) {
    options.emplace_back("onnxsim:unused_output", join(*unused_output));
  }

  auto* props = model.mutable_metadata_props();
  google::protobuf::RepeatedPtrField<onnx::StringStringEntryProto> kept;
  for (auto& prop : *props) {
    if (prop.key().rfind("onnxsim:", 0) != 0) {
      *kept.Add() = std::move(prop);
    }
  }
  props->Swap(&kept);
  for (const auto& [key, value] : options) {
    auto* entry = props->Add();
    entry->set_key(key);
    entry->set_value(value);
  }
}

// Runs InferShapesOnGraph + OptimizeGraphFixed to their own inner fixed
// point directly on `g`, with no ModelProto conversion at either end -- the
// Graph-resident core shared by OptAndShape's ModelFn (which wraps this in
// one Import before and one Export after, see below) and the fully
// Graph-native outer Pipeline (which shares one Import/Export across the
// *whole* outer fixed point instead, see Simplify's !rewriter branch).
// Returns whether anything changed.
bool OptAndShapeOnGraph(onnx::Graph& g, bool optimize, bool shape_inference,
                        size_t fixed_point_iters) {
  // See OptAndShape's own doc comment for why these caches are scoped to
  // one call of this function rather than cleared every round underneath.
  onnx::optimization::ClearTensorContentDigestCache();
  onnx::optimization::ClearRawHashCache();
  using GraphFnChanged = std::function<bool(onnx::Graph&)>;
  bool any_changed = false;
  GraphFnChanged InferShapesOnGraphChanged =
      shape_inference ? GraphFnChanged([&any_changed](onnx::Graph& graph) {
        onnxsim::ProfiledScope scope("InferShapes");
        const bool c = onnx::InferShapesOnGraph(graph);
        any_changed |= c;
        return c;
      })
                      : GraphFnChanged([](onnx::Graph&) { return false; });
  GraphFnChanged OptimizeGraphChanged = [optimize,
                                         &any_changed](onnx::Graph& graph) {
    onnxsim::ProfiledScope scope("Optimize");
    onnxsim::RegisterCustomOptimizerPasses();
    if (optimize) {
      // Unset all-zero recurrent initial states (issue #314) so the
      // subgraph computing them becomes dead and the passes below can
      // remove it. Not reflected in the "changed" signal below, but that
      // is safe: see FixedPointFn's bool-returning overload comment -- any
      // resulting dead-end elimination is itself reflected in the report.
      EliminateZeroRnnInitialState(graph);
    }
    const bool prev = onnx::optimization::InitializersAsConstants();
    onnx::optimization::SetInitializersAsConstants(
        config.initializers_as_constants);
    std::map<std::string, unsigned int> report;
    onnx::optimization::OptimizeGraphFixed(graph, config.optimizer_passes,
                                           &report,
                                           /*clear_tensor_digest_cache=*/false);
    onnx::optimization::SetInitializersAsConstants(prev);
    bool changed = false;
    for (const auto& pass_count : report) {
      if (pass_count.second != 0) {
        changed = true;
        break;
      }
    }
    any_changed |= changed;
    return changed;
  };
  FixedPointFn(InferShapesOnGraphChanged, OptimizeGraphChanged,
               fixed_point_iters)(g);
  return any_changed;
}

onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output) {
  // Register onnxsim's own optimizer passes into onnxoptimizer's registry
  // before the pass list is built below: fuse_consecutive_mul,
  // fuse_mul_into_conv, fuse_preceding_mul_into_conv, fuse_rms_norm and
  // eliminate_reshape_around_elementwise are auto-selected via
  // GetFuseAndEliminationPass, and fuse_matmul_add_bias_into_gemm_batched is
  // named explicitly. The call is idempotent.
  onnxsim::RegisterCustomOptimizerPasses();
  // Make shape inference aware of ONNX Runtime's quantized contrib operators
  // (QLinearAdd and friends) so shape deduction does not stop at them.
  onnxsim::RegisterContribOpSchemas();
  // Correct the determinism metadata of ops ONNX mis-annotates (e.g. Range) so
  // constant folding does not skip them.
  FixupSchemaDeterminism();
  // Register permissive placeholder schemas for custom ops exported into the
  // default ONNX domain (e.g. TensorRT plugins such as BatchedNMS_TRT) so the
  // checker below does not reject the model (GitHub issues #107, #220).
  RegisterCustomDefaultDomainOpSchemas(model);

  Check(model);

  config.tensor_size_threshold = tensor_size_threshold;
  config.initializers_as_constants = initializers_as_constants;
  config.optimizer_passes.clear();
  // onnxsim already folds ``Gather`` on a ``Shape`` into constants on its own:
  // ``_EvalPartialShape`` (above) plus data-propagating shape inference resolve
  // the statically-known dimensions and leave the genuinely dynamic ones
  // untouched. The onnx-optimizer ``eliminate_shape_gather`` pass is therefore
  // redundant here, and it aborts the whole process on graphs where a Gather
  // index cannot be statically resolved to an axis (common in dynamic-shape
  // detection models such as FasterRCNN). Always drop it from the pass list.
  std::vector<std::string> always_disabled_passes = {"eliminate_shape_gather"};
  // When initializers are treated as non-constant, keep ``Constant`` nodes in
  // producer form: ``extract_constant_to_initializer`` would rewrite every
  // Constant into an initializer, which -- being non-constant now -- would then
  // block onnxsim's own constant folding of genuinely-constant subgraphs. The
  // value-baking passes themselves already leave initializer weights alone via
  // ``IsConstantTensor`` (see the onnx-optimizer changes), so only this
  // representation-changing pass needs dropping.
  if (!initializers_as_constants) {
    always_disabled_passes.push_back("extract_constant_to_initializer");
  }
  auto is_disabled = [](const std::vector<std::string>& list,
                        const std::string& pass) {
    return std::find(list.begin(), list.end(), pass) != list.end();
  };
  // onnxoptimizer's common-subexpression / duplicate-initializer passes hash
  // tensor values and crash with "no supported data type: <N>" on element
  // types they cannot hash, such as the float8 zero points produced by
  // NVIDIA ModelOpt fp8 QDQ models (GitHub issue #348). When such a tensor
  // is present, transparently skip those two passes so the rest of
  // simplification still runs, instead of crashing outright. Only relevant
  // when some optimizer is actually going to run (skip_optimizers ==
  // nullopt already disables all of them).
  if (skip_optimizers && GraphHasCSEUnhashableTensor(model.graph())) {
    static const std::vector<std::string> kTensorValueHashingOptimizers = {
        "eliminate_common_subexpression", "eliminate_duplicate_initializer"};
    for (const auto& opt : kTensorValueHashingOptimizers) {
      if (!is_disabled(*skip_optimizers, opt)) {
        skip_optimizers->push_back(opt);
      }
    }
  }
  // skip_optimizers == nullopt means skiping all optimizers, so
  // config.optimizer_passes is empty
  if (skip_optimizers) {
    std::vector<std::string> passes;
    const auto all_passes = onnx::optimization::GetFuseAndEliminationPass();
    for (const auto& pass : all_passes) {
      if (!is_disabled(*skip_optimizers, pass) &&
          !is_disabled(always_disabled_passes, pass)) {
        passes.push_back(pass);
      }
    }
    // Opt into the batched MatMul+bias -> Gemm rewrite. onnx-optimizer
    // registers it as PassType::Other (so it is absent from
    // GetFuseAndEliminationPass), because it is a graph-shape rewrite rather
    // than a pure node reduction. Transformer linear layers apply a 2-D weight
    // to rank-3 activations, which the 2-D-only
    // ``fuse_matmul_add_bias_into_gemm`` cannot fuse; converting them to
    // ``Gemm`` lets runtimes dispatch their tuned GEMM kernels. The reshape
    // scaffolding it introduces around chains of element-wise ops is then
    // cancelled by ``eliminate_reshape_around_elementwise`` (a Nop pass in the
    // default set), which keeps the Gemms but drops the now-inverse reshape
    // pairs so the node count does not regress. Still honour an explicit
    // ``--skip-optimization`` for it.
    const std::string batched_gemm = "fuse_matmul_add_bias_into_gemm_batched";
    if (!is_disabled(*skip_optimizers, batched_gemm) &&
        !is_disabled(always_disabled_passes, batched_gemm)) {
      passes.push_back(batched_gemm);
    }
    config.optimizer_passes = passes;
  }

  // Every transform mutates the model in place.
  using ModelFn = std::function<void(onnx::ModelProto&)>;
  ModelFn FoldConstant;
  if (constant_folding) {
    FoldConstant = [&executor](onnx::ModelProto& model) {
      // Partial shape evaluation (issue #139) turns Shape/Gather-on-shape into
      // constants that the ordinary constant folder can then propagate.
      _EvalPartialShape(model);
      model = _FoldConstant(executor, std::move(model));
    };
  } else {
    FoldConstant = Identity;
  }
  // ``perform_optimization=False`` (skip_optimizers == nullopt) means the
  // caller wants the graph structure left alone, so the state elimination --
  // which relies on dead-end elimination to clean up behind it -- runs with the
  // optimizer or not at all.
  const bool optimize = skip_optimizers.has_value();

  int fixed_point_iters =
      std::getenv("ONNXSIM_FIXED_POINT_ITERS")
          ? std::atoi(std::getenv("ONNXSIM_FIXED_POINT_ITERS"))
          : 50;

  // Optionally profile every fixed-point function. Turned on by pointing
  // ``ONNXSIM_PROFILE`` at an output file (``ONNXSIM_PROFILE=1`` uses the
  // default name), mirroring ``ONNXSIM_FIXED_POINT_ITERS`` so it works from
  // every binding without a signature change. ``Profiled`` wraps a transform so
  // each invocation records its wall/CPU duration and peak RSS; the wrappers
  // nest, so the fixed points show up as parent spans of the transforms they
  // drive. When profiling is off ``Profiled`` returns the function unchanged
  // and
  // ``ProfiledScope`` is a no-op, so there is zero overhead.
  if (const char* profile_env = std::getenv("ONNXSIM_PROFILE")) {
    std::string profile_path = profile_env;
    if (profile_path == "1" || profile_path == "true" || profile_path == "on" ||
        profile_path == "yes") {
      profile_path = "onnxsim_profile.json";
    }
    if (!profile_path.empty()) {
      onnxsim::Profiler::Instance().Enable(profile_path);
    }
  }
  // Optionally break the per-round Optimize() cost down further, into each
  // PredicateBasedPass's "matching" (patternMatchPredicate) vs "modifying"
  // (runTransform) phase, per pass name -- see onnxoptimizer/pass.h's
  // PassPhaseTiming for scope and overhead. Exploratory diagnostic for
  // onnxsim issue #633; the summary prints to stderr after Pipeline runs.
  const bool profile_pass_phases = [] {
    const char* env = std::getenv("ONNXSIM_PROFILE_PASS_PHASES");
    return env != nullptr && std::string(env) != "0" &&
           std::string(env) != "false";
  }();
  if (profile_pass_phases) {
    onnx::optimization::ResetPassPhaseTimings();
    onnx::optimization::ResetPassTotalTimings();
    onnx::optimization::ResetCSEHashCompareTiming();
    onnx::optimization::ResetCSEPassTiming();
    onnx::optimization::ResetDeadendPassTiming();
    onnx::optimization::SetPassPhaseProfilingEnabled(true);
  }
  // Optionally merge ONNX Runtime's own per-session profiles into the onnxsim
  // trace -- the binding-agnostic counterpart of Python's
  // ``merge_ort_profile``, so the C ABI, Rust and WASM bindings can produce one
  // unified trace too. Enable the profiler if it is not already (there must be
  // a trace to merge into), then have it collect and splice ONNX Runtime's
  // traces at Finish().
  if (const char* merge_env = std::getenv("ONNXSIM_MERGE_ORT_PROFILE")) {
    std::string v = merge_env;
    if (!v.empty() && v != "0" && v != "false" && v != "off" && v != "no") {
      if (!onnxsim::Profiler::Instance().enabled()) {
        onnxsim::Profiler::Instance().Enable("onnxsim_profile.json");
      }
      onnxsim::Profiler::Instance().SetMergeOrtTraces(true);
    }
  }
  // Ensure the trace is flushed and the sampler thread is stopped even if a
  // transform throws mid-pipeline. Finish() is idempotent, so the explicit call
  // on the success path below is a harmless no-op the second time.
  struct ProfilerFinishGuard {
    ~ProfilerFinishGuard() { onnxsim::Profiler::Instance().Finish(); }
  } profiler_finish_guard;
  auto Profiled = [](const char* name, ModelFn fn) -> ModelFn {
    if (!onnxsim::Profiler::Instance().enabled()) {
      return fn;
    }
    return [name, fn = std::move(fn)](onnx::ModelProto& model) {
      onnxsim::ProfiledScope scope(name);
      fn(model);
    };
  };
  // Fully Graph-resident OptAndShape, built on onnx::InferShapesOnGraph
  // (onnx/common/graph_shape_inference.h) and onnx-optimizer's
  // OptimizeGraphFixed (onnxoptimizer/optimize.h): both run directly against
  // one onnx::Graph held for the whole inner fixed point, so this does
  // exactly one Import and one Export for the *entire* fixed point, however
  // many rounds it takes to converge -- instead of one Import+Export per
  // round. This is onnxsim issue #633's "remaining option 2" (eliminating
  // the round trip itself, not just its per-round cost); see #637 and #638
  // for the validation, profiling and follow-up fix that took this from
  // opt-in to the default.
  //
  // Remaining known scope gap: InferShapesOnGraph's own v1 scope leaves
  // control-flow ops (If/Loop/Scan) and function-body ops uninferred (safe:
  // exactly as if they had no registered schema, never wrong -- see that
  // function's doc comment), which can converge to less complete shape info
  // than a from-scratch protobuf InferShapes call for models using those
  // ops.
  ModelFn OptAndShape = Profiled(
      "OptAndShape",
      [optimize, shape_inference, fixed_point_iters](onnx::ModelProto& model) {
        std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
        if (g.get() == nullptr) {
          // Same fallback as Optimizer::optimize(): if we can't parse the
          // model, leave it untouched.
          return;
        }
        // Scope onnx-optimizer's tensor-hash caches (TensorContentDigest,
        // the typed-field path, and CSETensorHash's raw_data-branch cache
        // -- see onnxoptimizer/passes/{tensor_content_hash,cse_util}.h) to
        // this whole OptAndShape call, instead of the OptimizeGraphFixed
        // call below clearing them every round (its own default): `g` is
        // one resident Graph held across every round of the fixed point
        // underneath, and no round mutates a retained tensor's content in
        // place (only replaces it wholesale, which mints its own fresh
        // tensor_id() and simply misses these caches) -- so a hash computed
        // for a tensor that survives unchanged from one round to the next
        // stays valid, and correct, letting eliminate_duplicate_initializer
        // and eliminate_common_subexpression skip rehashing it on every one
        // of the ~dozens of rounds a deep repeated-block model can take.
        // Dominated in practice by the raw_data cache: real exported models
        // are raw_data-heavy, and measured 98%+ of those hash calls were
        // otherwise recomputing a value already seen earlier in the same
        // run (see onnxsim issue #633).
        OptAndShapeOnGraph(*g, optimize, shape_inference, fixed_point_iters);
        onnx::ModelProto out = onnx::PrepareOutput(model);
        onnx::ExportModelProto(&out, g, /*consume_tensor_data=*/true);
        // OptimizeGraphFixed never sees model-local functions (they live
        // on ModelProto, not Graph), so carry them over unchanged --
        // mirroring Optimizer::optimize()'s own AddFunctionsToModel.
        for (const auto& function_proto : model.functions()) {
          *out.add_functions() = function_proto;
        }
        model = std::move(out);
      });
  bool converged = false;
  // Set from `model.ir_version()` right before `Pipeline` runs at the bottom
  // of this function -- fold's throwaway sub-models need it (see
  // RunOpsOnGraph's own comment). Declared here, alongside `converged`, and
  // not inside the `else` branch below that builds `Pipeline`'s closures:
  // those closures (and the reference to this variable they capture) are
  // still called from `Pipeline(sim_model)` well after that branch's own
  // scope -- and this variable's lifetime, were it declared there instead --
  // has ended.
  int64_t fold_ir_version = 0;
  ModelFn Pipeline;
  if (rewriter) {
    // Run the user-supplied rewriter (e.g. an onnxscript.rewriter rule set) as
    // the outermost stage of the fixed point: each round drives shape
    // inference, onnxoptimizer and constant folding to a fixed point, then
    // rewrites, then repeats until the whole thing stops changing. This lets a
    // rewrite expose new optimizer/folding opportunities and vice versa. The
    // rewriter runs on the ``ModelProto`` directly, so no internal-graph
    // conversion is involved.
    ModelFn OptAndShapeAndFold = Profiled(
        "OptAndShapeAndFold",
        FixedPointFn(OptAndShape, Profiled("FoldConstant", FoldConstant),
                     fixed_point_iters));
    ModelFn RewriteInPlace =
        Profiled("Rewrite", [rewriter](onnx::ModelProto& model) {
          // ``_Run`` rewrites in place and returns whether it changed anything;
          // when it reports no change it leaves ``model`` untouched, so no copy
          // is made. The fixed point's fingerprint comparison then sees the
          // unchanged model and converges without an extra ModelProto
          // round-trip.
          rewriter->_Run(model);
        });
    Pipeline =
        Profiled("Pipeline", FixedPointFn(OptAndShapeAndFold, RewriteInPlace,
                                          fixed_point_iters, &converged));
  } else {
    // Fully Graph-resident outer pipeline: OptAndShapeOnGraph and the
    // Graph-native constant folder (_EvalPartialShapeOnGraph +
    // _FoldConstantOnGraph) both run directly against one onnx::Graph held
    // for the *entire* outer fixed point, so this whole Simplify() call
    // does exactly one Import and one Export -- instead of one Import+
    // Export per outer round, which #637/#638 already eliminated at the
    // *inner* (OptAndShape) level but which this pipeline still paid once
    // per outer round until now. Only available without a rewriter:
    // onnxscript's rewriter only understands ModelProto (see the `if
    // (rewriter)` branch above), so a supplied rewriter still needs the
    // ModelProto-based OptAndShape/FoldConstant path.
    using GraphFn = std::function<void(onnx::Graph&)>;
    using GraphFnChanged = std::function<bool(onnx::Graph&)>;
    GraphFnChanged OptAndShapeOnGraphChanged =
        [optimize, shape_inference, fixed_point_iters](onnx::Graph& graph) {
          onnxsim::ProfiledScope scope("OptAndShape");
          return OptAndShapeOnGraph(graph, optimize, shape_inference,
                                    fixed_point_iters);
        };
    // `fold_ir_version` is declared above, alongside `converged`: see its own
    // comment for why it cannot live in this block despite only being read
    // here and set (from `model.ir_version()`) in the `Pipeline` lambda
    // below.
    GraphFnChanged FoldConstantOnGraphChanged =
        constant_folding
            ? GraphFnChanged([&executor, &fold_ir_version](onnx::Graph& graph) {
                onnxsim::ProfiledScope scope("FoldConstant");
                const bool a = _EvalPartialShapeOnGraph(graph);
                const bool b =
                    _FoldConstantOnGraph(executor, graph, fold_ir_version);
                return a || b;
              })
            : GraphFnChanged([](onnx::Graph&) { return false; });
    GraphFn PipelineOnGraph =
        FixedPointFn(OptAndShapeOnGraphChanged, FoldConstantOnGraphChanged,
                     fixed_point_iters, &converged);
    Pipeline = Profiled("Pipeline", [PipelineOnGraph, &fold_ir_version](
                                        onnx::ModelProto& model) {
      std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
      if (g.get() == nullptr) {
        // Same fallback as Optimizer::optimize(): if we can't parse the
        // model, leave it untouched.
        return;
      }
      fold_ir_version = model.ir_version();
      PipelineOnGraph(*g);
      onnx::ModelProto out = onnx::PrepareOutput(model);
      onnx::ExportModelProto(&out, g, /*consume_tensor_data=*/true);
      // OptimizeGraphFixed never sees model-local functions (they live on
      // ModelProto, not Graph), so carry them over unchanged -- mirroring
      // Optimizer::optimize()'s own AddFunctionsToModel.
      for (const auto& function_proto : model.functions()) {
        *out.add_functions() = function_proto;
      }
      model = std::move(out);
    });
  }
  // The fixed points mutate in place, so make one working copy of the (const)
  // input model and simplify it in place.
  onnx::ModelProto sim_model = model;
  // Matches onnx_simplifier.py's default (mutable_initializer=False): fold
  // initializers that also appear as graph inputs like any other constant.
  if (!mutable_initializer) {
    RemoveInitializerFromInput(sim_model);
  }
  // Overwrite named input shapes, if requested, before shape inference or
  // any transform runs -- mirrors onnx_simplifier.py's overwrite_input_shapes
  // loop, applied here instead of by the Python wrapper.
  if (overwrite_input_shapes) {
    ApplyInputShapeOverwrite(sim_model, *overwrite_input_shapes);
  }
  // Drop the named graph outputs, if requested, before the fixed point so
  // dead-end elimination cleans up nodes that only fed them.
  if (unused_output) {
    RemoveUnusedOutputs(sim_model, *unused_output);
  }
  // Optionally inline the model's local (model-defined) functions into the main
  // graph up front, so the optimizer, shape inference and constant folding
  // below see through them into a flat op graph. Done before the opset
  // conversion and fixed point so everything downstream operates on the inlined
  // graph; schema-defined functions are left untouched. Off by default, so a
  // model with functions is otherwise simplified exactly as before.
  if (include_inline_functions) {
    onnx::inliner::InlineLocalFunctions(sim_model);
  }
  // Optionally convert the model to a different opset version (of the default
  // ONNX domain) first, so the simplification below can clean up any redundant
  // nodes the version converter introduces.
  if (target_opset_version) {
    sim_model = ConvertOpsetVersion(sim_model, *target_opset_version);
  }
  {
    // A single root span so the profiled fixed points nest under one box in the
    // flame graph (a no-op when profiling is disabled).
    onnxsim::ProfiledScope root("Simplify");
    Pipeline(sim_model);
  }
  // Flush the profiling trace and print the per-function summary. Safe to call
  // unconditionally: Finish() is a no-op unless profiling was enabled above.
  onnxsim::Profiler::Instance().Finish();
  if (profile_pass_phases) {
    onnx::optimization::SetPassPhaseProfilingEnabled(false);
    std::vector<std::pair<std::string, onnx::optimization::PassPhaseTiming>>
        rows(onnx::optimization::GetPassPhaseTimings().begin(),
             onnx::optimization::GetPassPhaseTimings().end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return (a.second.match_ms + a.second.transform_ms) >
             (b.second.match_ms + b.second.transform_ms);
    });
    std::cerr << "\nonnxsim pass-phase profiling (PredicateBasedPass "
                 "matching vs modifying)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << std::left << std::setw(38) << "pass" << std::right
              << std::setw(12) << "match(ms)" << std::setw(10) << "calls"
              << std::setw(14) << "modify(ms)" << std::setw(10) << "calls"
              << std::setw(12) << "total(ms)" << "\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n";
    double total_match_ms = 0, total_transform_ms = 0;
    for (const auto& [name, t] : rows) {
      std::cerr << std::left << std::setw(38) << name << std::right
                << std::setw(12) << std::fixed << std::setprecision(2)
                << t.match_ms << std::setw(10) << t.match_calls << std::setw(14)
                << t.transform_ms << std::setw(10) << t.transform_calls
                << std::setw(12) << (t.match_ms + t.transform_ms) << "\n";
      total_match_ms += t.match_ms;
      total_transform_ms += t.transform_ms;
    }
    std::cerr << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "TOTAL matching: " << total_match_ms
              << "ms, TOTAL modifying: " << total_transform_ms << "ms\n";

    // Coarser companion table: total wall time inside EVERY pass's
    // runPass(Graph&) call (both kinds), which is what actually accounts
    // for the full Optimize() cost -- see PassTotalTiming's comment.
    std::vector<std::pair<std::string, onnx::optimization::PassTotalTiming>>
        total_rows(onnx::optimization::GetPassTotalTimings().begin(),
                   onnx::optimization::GetPassTotalTimings().end());
    std::sort(total_rows.begin(), total_rows.end(),
              [](const auto& a, const auto& b) {
                return a.second.total_ms > b.second.total_ms;
              });
    std::cerr << "\nonnxsim pass-phase profiling (total runPass() time, all "
                 "pass kinds)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << std::left << std::setw(38) << "pass" << std::right
              << std::setw(14) << "total(ms)" << std::setw(10) << "calls"
              << "\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n";
    double grand_total_ms = 0;
    for (const auto& [name, t] : total_rows) {
      std::cerr << std::left << std::setw(38) << name << std::right
                << std::setw(14) << std::fixed << std::setprecision(2)
                << t.total_ms << std::setw(10) << t.calls << "\n";
      grand_total_ms += t.total_ms;
    }
    std::cerr << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "GRAND TOTAL (sum of all pass runPass() calls): "
              << grand_total_ms << "ms\n";

    // CSETensorHash/CSETensorCompare breakdown -- the actual hashing and
    // equality-checking work inside eliminate_duplicate_initializer and
    // eliminate_common_subexpression's hash-map lookups. Splits raw_data
    // (real exported-model weights; a byte hash / a single memcmp) from
    // typed-field (BLAKE3-digest-backed; rarer in practice) so it's clear
    // which one, if either, actually accounts for those two passes' cost.
    const auto& cse_t = onnx::optimization::GetCSEHashCompareTiming();
    std::cerr << "\nonnxsim CSETensorHash / CSETensorCompare breakdown "
                 "(inside eliminate_duplicate_initializer / "
                 "eliminate_common_subexpression's hash-map lookups)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << std::left << std::setw(24) << "" << std::right << std::setw(14)
              << "total(ms)" << std::setw(10) << "calls" << "\n"
              << std::left << std::setw(24) << "raw_data hash" << std::right
              << std::setw(14) << std::fixed << std::setprecision(2)
              << cse_t.raw_hash_ms << std::setw(10) << cse_t.raw_hash_calls
              << "\n"
              << std::left << std::setw(24) << "typed-field hash" << std::right
              << std::setw(14) << cse_t.typed_hash_ms << std::setw(10)
              << cse_t.typed_hash_calls << "\n"
              << std::left << std::setw(24) << "raw_data compare" << std::right
              << std::setw(14) << cse_t.raw_compare_ms << std::setw(10)
              << cse_t.raw_compare_calls << "\n"
              << std::left << std::setw(24) << "typed-field compare"
              << std::right << std::setw(14) << cse_t.typed_compare_ms
              << std::setw(10) << cse_t.typed_compare_calls << "\n"
              << std::left << std::setw(24) << "node hash (CSENodeHash)"
              << std::right << std::setw(14) << cse_t.node_hash_ms
              << std::setw(10) << cse_t.node_hash_calls << "\n"
              << std::left << std::setw(24) << "  of which attrs+sort"
              << std::right << std::setw(14) << cse_t.node_hash_attrsort_ms
              << std::setw(10) << cse_t.node_hash_attrsort_calls << "\n"
              << std::left << std::setw(24) << "node equal (CSEEqual)"
              << std::right << std::setw(14) << cse_t.node_equal_ms
              << std::setw(10) << cse_t.node_equal_calls << "\n"
              << std::left << std::setw(24) << "  of which attrs+sort"
              << std::right << std::setw(14) << cse_t.node_equal_attrsort_ms
              << std::setw(10) << cse_t.node_equal_attrsort_calls << "\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "TOTAL hash: "
              << (cse_t.raw_hash_ms + cse_t.typed_hash_ms + cse_t.node_hash_ms)
              << "ms, TOTAL compare: "
              << (cse_t.raw_compare_ms + cse_t.typed_compare_ms +
                  cse_t.node_equal_ms)
              << "ms\n";

    // Actual cache hit rate: see cse_util.h's g_raw_hash_cache, keyed by
    // Tensor::tensor_id().
    if (cse_t.raw_hash_calls > 0) {
      std::cerr << "raw_data hash cache: " << cse_t.raw_hash_cache_hits << "/"
                << cse_t.raw_hash_calls << " calls ("
                << (100.0 * cse_t.raw_hash_cache_hits / cse_t.raw_hash_calls)
                << "%) were hits (" << cse_t.raw_hash_cache_hit_ms << "ms) vs "
                << cse_t.raw_hash_cache_misses << " misses ("
                << cse_t.raw_hash_cache_miss_ms << "ms).\n";
    }

    // eliminate_common_subexpression's own outer-loop breakdown: filtering
    // (hasUses/IsSupportedByCSE), the hash_map.emplace() lookup itself
    // (overlaps with node_hash_ms/node_equal_ms above -- same work, two
    // views), and replacing a found duplicate's uses.
    const auto& cse_pass_t = onnx::optimization::GetCSEPassTiming();
    std::cerr << "\nonnxsim eliminate_common_subexpression outer-loop "
                 "breakdown (across "
              << cse_pass_t.calls << " calls, " << cse_pass_t.nodes_seen
              << " nodes seen, " << cse_pass_t.nodes_filtered_out
              << " filtered out, " << cse_pass_t.nodes_replaced
              << " replaced)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "  filter (hasUses/IsSupportedByCSE): " << cse_pass_t.filter_ms
              << "ms\n"
              << "  lookup (hash_map.emplace):          "
              << cse_pass_t.lookup_ms << "ms\n"
              << "  replace (tryReplacingAllUsesWith):   "
              << cse_pass_t.replace_ms << "ms\n";

    // eliminate_deadend's own breakdown: the liveness check (hasUses) vs
    // actually unlinking/freeing a dead node (destroyCurrent).
    const auto& dead_t = onnx::optimization::GetDeadendPassTiming();
    std::cerr << "\nonnxsim eliminate_deadend breakdown (across "
              << dead_t.calls << " calls, " << dead_t.nodes_seen
              << " nodes seen, " << dead_t.nodes_removed << " removed)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "  hasUses():      " << dead_t.has_uses_ms << "ms\n"
              << "  destroyCurrent(): " << dead_t.destroy_ms << "ms\n";
  }
  // Simplification (and some onnx-optimizer passes) can leave nodes without a
  // name; assign unique names to them so downstream tools that key on node
  // names keep working (issue #269).
  AssignMissingNodeNames(sim_model);
  DropIncompleteValueInfo(sim_model);
  RecordSimplifyOptionsMetadata(sim_model, skip_optimizers, constant_folding,
                                shape_inference, tensor_size_threshold,
                                target_opset_version, initializers_as_constants,
                                include_inline_functions, mutable_initializer,
                                overwrite_input_shapes, unused_output);
  Check(sim_model);
  if (!converged) {
    std::cout << "WARNING: the simplification stopped because of timeout. "
                 "Please set environment variable `ONNXSIM_FIXED_POINT_ITERS` "
                 "to a number higher than "
              << fixed_point_iters << "if you want further simplification."
              << std::endl;
  }
  return sim_model;
}

void SimplifyPath(
    const ModelExecutor& executor, const std::string& in_path,
    const std::string& out_path,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output) {
  const bool debug_timing = std::getenv("ONNXSIM_DEBUG_PATH_TIMING") != nullptr;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto elapsed_ms = [](auto t0, auto t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  onnx::ModelProto model;
  {
    const auto t0 = now();
    onnx::optimization::loadModel(&model, in_path, true);
    if (debug_timing) {
      std::cerr << "SimplifyPath: loadModel " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }

  {
    const auto t0 = now();
    model =
        Simplify(executor, model, skip_optimizers, constant_folding,
                 shape_inference, tensor_size_threshold, target_opset_version,
                 rewriter, initializers_as_constants, include_inline_functions,
                 mutable_initializer, overwrite_input_shapes, unused_output);
    if (debug_timing) {
      std::cerr << "SimplifyPath: Simplify " << elapsed_ms(t0, now()) << "ms\n";
    }
  }

  // Prefer a single self-contained inline file over external data: onnx's
  // own checker (onnx.checker.check_model, called by every Python caller's
  // correctness check) is drastically slower validating an external-data
  // model than an equivalent inline one -- 18.6s vs ~4s measured on an
  // 833MB model, apparently from re-reading/re-validating the sibling data
  // file rather than working off the already-parsed in-memory tensors. Only
  // fall back to external data when the model does not fit in a single
  // protobuf message (the 2GB limit `saveModel`'s unconditional external-
  // data write used to sidestep unconditionally).
  constexpr size_t kProtobufSizeLimit = (size_t(2) << 30) - 9999;
  bool needs_external_data;
  {
    const auto t0 = now();
    needs_external_data = model.ByteSizeLong() >= kProtobufSizeLimit;
    if (debug_timing) {
      std::cerr << "SimplifyPath: ByteSizeLong " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }
  {
    const auto t0 = now();
    onnx::optimization::saveModel(&model, out_path, needs_external_data, "");
    if (debug_timing) {
      std::cerr << "SimplifyPath: saveModel " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }
}
