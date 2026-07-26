// Microbenchmark for the arena change in onnxsim/onnxsim.cpp (RunOp).
//
// RunOp builds a throwaway onnx::ModelProto sub-model for every foldable node --
// often thousands of times across a fixed-point run -- copying initializers and
// nodes into it, serializing it for the executor, then discarding it. Each
// sub-model is a deep tree of nested messages, so destroying it walks the whole
// tree freeing every sub-message individually. The change allocates that
// sub-model on a google::protobuf::Arena so its entire tree is released in one
// bulk free instead.
//
// This benchmark isolates exactly that cost. It deliberately does NOT go through
// onnxsim's ONNX Runtime executor: in the real pipeline each RunOp call also
// spins up a full Ort::Session, which dwarfs the sub-model teardown, so an
// end-to-end Simplify wall-time measurement drowns the arena delta in ORT noise.
// Here we build sub-models shaped like the ones RunOp produces and measure the
// build+destroy cost with and without the arena.
//
// Two measurements are reported:
//   [full cycle]  build + destroy one sub-model per iteration, a fresh arena per
//                 op -- exactly what RunOp does (arena vs a plain stack message).
//   [teardown]    build N sub-models first, then time only their destruction
//                 (vector<unique_ptr> clear vs a single Arena::Reset) -- the
//                 "destruction cost" the change is aimed at, in isolation.
//
// The sub-model shape is tunable so you can see how the win scales with the
// number of nested messages (which is what arena teardown collapses -- raw_data
// byte buffers are heap strings freed the same way in both variants):
//
//   argv: [iters] [inits_per_model] [nodes_per_model] [dim]
//   defaults: 20000 8 1 16
//
// Build & run: see bench/fold_teardown_bench.sh (points -I/-L at your onnx +
// protobuf, which are already built when you build onnxsim).

#include <google/protobuf/arena.h>
#include <onnx/onnx_pb.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using onnx::ModelProto;
using Clock = std::chrono::steady_clock;

namespace {

// Build a sub-model shaped like the one RunOp assembles: an ir_version, one
// opset import, `inits` constant initializers (each also declared as a graph
// input, as RunOp does for the operands it copies in), and `nodes` nodes that
// consume them. `sink` accumulates a cheap value so the optimizer cannot elide
// the construction.
void FillModel(ModelProto& m, int inits, int nodes, int dim,
               uint64_t& sink) {
  m.set_ir_version(10);
  auto* opset = m.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);

  auto* g = m.mutable_graph();
  g->set_name("g");
  // Small raw_data on purpose: the point is the count of nested *messages*, not
  // byte volume. raw_data is a heap std::string freed identically in both
  // variants, so inflating it would only add common-mode noise.
  const std::string raw(static_cast<size_t>(dim) * 4, '\0');
  for (int i = 0; i < inits; i++) {
    const std::string name = "W" + std::to_string(i);
    auto* t = g->add_initializer();
    t->set_name(name);
    t->set_data_type(onnx::TensorProto::FLOAT);
    t->add_dims(dim);
    t->set_raw_data(raw);
    auto* vi = g->add_input();
    vi->set_name(name);
  }
  for (int i = 0; i < nodes; i++) {
    auto* n = g->add_node();
    n->set_op_type("Sum");
    n->set_name("n" + std::to_string(i));
    for (int j = 0; j < inits; j++) {
      n->add_input("W" + std::to_string(j));
    }
    n->add_output("h" + std::to_string(i));
  }
  auto* out = g->add_output();
  out->set_name(nodes > 0 ? "h0" : "W0");

  sink += static_cast<uint64_t>(g->initializer_size()) + g->node_size();
}

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 20000;
  const int inits = argc > 2 ? std::atoi(argv[2]) : 8;
  const int nodes = argc > 3 ? std::atoi(argv[3]) : 1;
  const int dim = argc > 4 ? std::atoi(argv[4]) : 16;

  std::printf(
      "iters=%d inits_per_model=%d nodes_per_model=%d dim=%d "
      "(nested messages/model ~= %d)\n",
      iters, inits, nodes, dim, inits /*init*/ + inits /*input vi*/ +
                                    nodes /*node*/ + 1 /*output vi*/ +
                                    1 /*opset*/);

  volatile uint64_t global_sink = 0;

  // ---- [full cycle] build + destroy one sub-model per op, as RunOp does ---- //
  {
    uint64_t sink = 0;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
      ModelProto m;
      FillModel(m, inits, nodes, dim, sink);
    }
    auto t1 = Clock::now();
    global_sink += sink;

    sink = 0;
    auto t2 = Clock::now();
    for (int i = 0; i < iters; i++) {
      google::protobuf::Arena arena;
      auto* m = google::protobuf::Arena::CreateMessage<ModelProto>(&arena);
      FillModel(*m, inits, nodes, dim, sink);
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
    std::vector<std::unique_ptr<ModelProto>> heap_models;
    heap_models.reserve(iters);
    for (int i = 0; i < iters; i++) {
      auto m = std::make_unique<ModelProto>();
      FillModel(*m, inits, nodes, dim, sink);
      heap_models.push_back(std::move(m));
    }
    auto h0 = Clock::now();
    heap_models.clear();  // recursive per-message destruction
    auto h1 = Clock::now();

    google::protobuf::Arena arena;
    std::vector<ModelProto*> arena_models;
    arena_models.reserve(iters);
    for (int i = 0; i < iters; i++) {
      auto* m = google::protobuf::Arena::CreateMessage<ModelProto>(&arena);
      FillModel(*m, inits, nodes, dim, sink);
      arena_models.push_back(m);
    }
    auto a0 = Clock::now();
    arena.Reset();  // one bulk free
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
