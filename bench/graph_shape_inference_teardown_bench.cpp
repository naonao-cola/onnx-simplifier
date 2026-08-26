// Microbenchmark for the arena change in graph_shape_inference.cc (ProcessNode).
//
// ProcessNode builds a throwaway NodeProto (the node's attribute tree, including
// a full subgraph body export for If/Loop/Scan) plus per-input TypeProto/
// TensorProto adapters, for every node visited -- and Run() drives
// InferShapesOnGraph to a fixed point over the whole graph, so this happens many
// times per node across a typical simplification run. Each of these is a small
// tree of nested protobuf messages, so destroying it walks the tree freeing
// every sub-message individually. The change allocates the whole per-visit tree
// on a google::protobuf::Arena so it is released in one bulk free instead.
//
// This benchmark isolates exactly that cost, the same way bench/
// fold_teardown_bench.cpp isolates RunOp's arena change from ONNX Runtime
// session-creation noise: an end-to-end Simplify measurement mixes in schema
// lookup, symbol-table bookkeeping and the actual shape/type inference
// functions, which drown the message-construction delta. Here we build
// NodeProto/TypeProto/TensorProto trees shaped like the ones ProcessNode
// produces and measure the build+destroy cost with and without the arena.
//
// Two measurements are reported:
//   [full cycle]  build + destroy one node-visit's messages per iteration, a
//                 fresh arena per visit -- exactly what ProcessNode does (arena
//                 vs plain heap-allocated messages).
//   [teardown]    build N node-visits first, then time only their destruction
//                 (vector<unique_ptr>/vector clear vs a single Arena::Reset) --
//                 the "destruction cost" the change is aimed at, in isolation.
//
// The node shape is tunable so you can see how the win scales with the number
// of nested messages:
//
//   argv: [iters] [num_inputs] [num_outputs] [num_attrs] [rank] [num_const_inputs]
//   defaults: 20000 3 1 2 4 1
//
// Build & run: see bench/graph_shape_inference_teardown_bench.sh (points
// -I/-L at your onnx + protobuf, which are already built when you build
// onnxsim).

#include <google/protobuf/arena.h>
#include <onnx/onnx_pb.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using onnx::AttributeProto_AttributeType_INT;
using onnx::NodeProto;
using onnx::TensorProto;
using onnx::TypeProto;
using Clock = std::chrono::steady_clock;

namespace {

// One "node visit"'s worth of throwaway messages: a NodeProto shell (mirroring
// ProcessNode's np, including a handful of int attributes -- AttributeProto is
// itself a nested message, same as a real op's attributes) plus one TypeProto
// per input (mirroring EncodeCurrentType's tensor_type/shape/dim fill) and,
// for `num_const_inputs` of them, a small TensorProto (mirroring encodeTensor
// for a Reshape-shape-sized constant input). `sink` accumulates a cheap value
// so the optimizer cannot elide the construction.
struct NodeVisit {
  NodeProto* np;
  std::vector<TypeProto*> input_types;
  std::vector<TensorProto*> input_data;
};

template <typename NodeProtoFactory, typename TypeProtoFactory,
          typename TensorProtoFactory>
void FillNodeVisit(int num_inputs, int num_outputs, int num_attrs, int rank,
                    int num_const_inputs, uint64_t& sink,
                    NodeProtoFactory make_node, TypeProtoFactory make_type,
                    TensorProtoFactory make_tensor, NodeProto*& out_np,
                    std::vector<TypeProto*>& out_types,
                    std::vector<TensorProto*>& out_data) {
  NodeProto* np = make_node();
  np->set_op_type("Foo");
  for (int i = 0; i < num_inputs; i++) {
    np->add_input("in" + std::to_string(i));
  }
  for (int i = 0; i < num_outputs; i++) {
    np->add_output("out" + std::to_string(i));
  }
  for (int i = 0; i < num_attrs; i++) {
    auto* attr = np->add_attribute();
    attr->set_name("a" + std::to_string(i));
    attr->set_type(AttributeProto_AttributeType_INT);
    attr->set_i(i);
  }
  out_np = np;

  out_types.reserve(static_cast<size_t>(num_inputs));
  out_data.reserve(static_cast<size_t>(num_const_inputs));
  for (int i = 0; i < num_inputs; i++) {
    TypeProto* t = make_type();
    auto* tensor_type = t->mutable_tensor_type();
    tensor_type->set_elem_type(1); // FLOAT
    auto* shape = tensor_type->mutable_shape();
    for (int d = 0; d < rank; d++) {
      shape->add_dim()->set_dim_value(d + 1);
    }
    out_types.push_back(t);

    if (i < num_const_inputs) {
      TensorProto* tp = make_tensor();
      tp->set_data_type(TensorProto::INT64);
      tp->add_dims(rank);
      tp->set_raw_data(std::string(static_cast<size_t>(rank) * 8, '\0'));
      out_data.push_back(tp);
    }
  }
  sink += static_cast<uint64_t>(np->attribute_size()) + out_types.size() +
          out_data.size();
}

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 20000;
  const int num_inputs = argc > 2 ? std::atoi(argv[2]) : 3;
  const int num_outputs = argc > 3 ? std::atoi(argv[3]) : 1;
  const int num_attrs = argc > 4 ? std::atoi(argv[4]) : 2;
  const int rank = argc > 5 ? std::atoi(argv[5]) : 4;
  const int num_const_inputs = argc > 6 ? std::atoi(argv[6]) : 1;

  std::printf(
      "iters=%d num_inputs=%d num_outputs=%d num_attrs=%d rank=%d "
      "num_const_inputs=%d (nested messages/visit ~= %d)\n",
      iters, num_inputs, num_outputs, num_attrs, rank, num_const_inputs,
      1 /*NodeProto*/ + num_attrs /*AttributeProto*/ +
          num_inputs /*TypeProto*/ + num_inputs /*TensorShapeProto*/ +
          num_inputs * rank /*Dimension*/ + num_const_inputs /*TensorProto*/);

  volatile uint64_t global_sink = 0;

  // ---- [full cycle] build + destroy one node visit per op, as ProcessNode does --- //
  {
    uint64_t sink = 0;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
      NodeProto np;
      std::vector<TypeProto> input_types(static_cast<size_t>(num_inputs));
      std::vector<TensorProto> input_data;
      input_data.reserve(static_cast<size_t>(num_inputs));

      NodeProto* out_np = nullptr;
      std::vector<TypeProto*> out_types;
      std::vector<TensorProto*> out_data;
      size_t type_idx = 0;
      FillNodeVisit(
          num_inputs, num_outputs, num_attrs, rank, num_const_inputs, sink,
          [&] { return &np; },
          [&] { return &input_types[type_idx++]; },
          [&] {
            input_data.emplace_back();
            return &input_data.back();
          },
          out_np, out_types, out_data);
    }
    auto t1 = Clock::now();
    global_sink += sink;

    sink = 0;
    auto t2 = Clock::now();
    for (int i = 0; i < iters; i++) {
      google::protobuf::Arena arena;
      NodeProto* np = google::protobuf::Arena::Create<NodeProto>(&arena);
      google::protobuf::RepeatedPtrField<TypeProto> input_types(&arena);
      google::protobuf::RepeatedPtrField<TensorProto> input_data(&arena);
      input_types.Reserve(num_inputs);
      input_data.Reserve(num_inputs);

      NodeProto* out_np = nullptr;
      std::vector<TypeProto*> out_types;
      std::vector<TensorProto*> out_data;
      FillNodeVisit(
          num_inputs, num_outputs, num_attrs, rank, num_const_inputs, sink,
          [&] { return np; }, [&] { return input_types.Add(); },
          [&] { return input_data.Add(); }, out_np, out_types, out_data);
    }
    auto t3 = Clock::now();
    global_sink += sink;

    const double heap = ms(t0, t1), aren = ms(t2, t3);
    std::printf(
        "[full cycle]  heap=%8.2f ms (%6.3f us/op)   arena=%8.2f ms "
        "(%6.3f us/op)   speedup=%.2fx\n",
        heap, 1000.0 * heap / iters, aren, 1000.0 * aren / iters, heap / aren);
  }

  // ---- [teardown] build everything first, then time destruction only ------ //
  {
    uint64_t sink = 0;
    struct HeapVisit {
      std::unique_ptr<NodeProto> np;
      std::vector<TypeProto> input_types;
      std::vector<TensorProto> input_data;
    };
    std::vector<HeapVisit> heap_visits;
    heap_visits.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; i++) {
      HeapVisit v;
      v.np = std::make_unique<NodeProto>();
      v.input_types.resize(static_cast<size_t>(num_inputs));
      v.input_data.reserve(static_cast<size_t>(num_inputs));

      NodeProto* out_np = nullptr;
      std::vector<TypeProto*> out_types;
      std::vector<TensorProto*> out_data;
      size_t type_idx = 0;
      NodeProto* np_ptr = v.np.get();
      FillNodeVisit(
          num_inputs, num_outputs, num_attrs, rank, num_const_inputs, sink,
          [&] { return np_ptr; },
          [&, ptr = &v.input_types] { return &(*ptr)[type_idx++]; },
          [&, ptr = &v.input_data] {
            ptr->emplace_back();
            return &ptr->back();
          },
          out_np, out_types, out_data);
      heap_visits.push_back(std::move(v));
    }
    auto h0 = Clock::now();
    heap_visits.clear(); // recursive per-message destruction
    auto h1 = Clock::now();

    google::protobuf::Arena arena;
    std::vector<NodeProto*> arena_visits;
    arena_visits.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; i++) {
      // Local (stack) RepeatedPtrField wrapping the shared arena: elements
      // added via Add() live on `arena` and stay reachable through the raw
      // pointers collected below regardless of this container's own scope --
      // exactly the pattern ProcessNode itself uses.
      NodeProto* np = google::protobuf::Arena::Create<NodeProto>(&arena);
      google::protobuf::RepeatedPtrField<TypeProto> input_types(&arena);
      google::protobuf::RepeatedPtrField<TensorProto> input_data(&arena);
      input_types.Reserve(num_inputs);
      input_data.Reserve(num_inputs);

      NodeProto* out_np = nullptr;
      std::vector<TypeProto*> out_types;
      std::vector<TensorProto*> out_data;
      FillNodeVisit(
          num_inputs, num_outputs, num_attrs, rank, num_const_inputs, sink,
          [&] { return np; }, [&] { return input_types.Add(); },
          [&] { return input_data.Add(); }, out_np, out_types, out_data);
      arena_visits.push_back(np);
    }
    auto a0 = Clock::now();
    arena.Reset(); // one bulk free
    auto a1 = Clock::now();
    global_sink += sink;

    const double heap = ms(h0, h1), aren = ms(a0, a1);
    std::printf(
        "[teardown]    heap=%8.2f ms (%6.3f us/op)   arena=%8.2f ms "
        "(%6.3f us/op)   speedup=%.2fx\n",
        heap, 1000.0 * heap / iters, aren, 1000.0 * aren / iters,
        aren > 0 ? heap / aren : 0.0);
  }

  std::fprintf(stderr, "sink=%llu\n",
               static_cast<unsigned long long>(global_sink));
  return 0;
}
