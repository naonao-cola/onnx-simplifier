/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "profiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

// Platform-specific resident-set-size readers. Everything degrades to 0 (peak
// memory simply unreported) on platforms none of these branches cover, so the
// timing side of the profiler keeps working regardless.
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(_WIN32)
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#endif

namespace onnxsim {

namespace {

// Current resident set size of this process, in bytes. Returns 0 when it cannot
// be determined.
size_t ReadCurrentRssBytes() {
#if defined(__linux__)
  // /proc/self/statm reports memory in pages: "size resident shared ...".
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (f == nullptr) {
    return 0;
  }
  long size_pages = 0;
  long resident_pages = 0;
  const int matched = std::fscanf(f, "%ld %ld", &size_pages, &resident_pages);
  std::fclose(f);
  if (matched != 2) {
    return 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  return static_cast<size_t>(resident_pages) * static_cast<size_t>(page_size);
#elif defined(__APPLE__)
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<size_t>(info.resident_size);
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return static_cast<size_t>(pmc.WorkingSetSize);
  }
  return 0;
#else
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    // ru_maxrss is kilobytes on Linux/BSD; this branch is only the fallback for
    // platforms without a live-RSS source, so a peak high-water mark is the best
    // available approximation.
    return static_cast<size_t>(ru.ru_maxrss) * 1024;
  }
  return 0;
#endif
}

// Process CPU time consumed so far (user + system, across all threads), in
// microseconds. Uses CLOCK_PROCESS_CPUTIME_ID where available and falls back to
// std::clock().
uint64_t CpuMicros() {
#if defined(CLOCK_PROCESS_CPUTIME_ID)
  struct timespec ts;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0) {
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ull +
           static_cast<uint64_t>(ts.tv_nsec) / 1000ull;
  }
#endif
  return static_cast<uint64_t>(static_cast<double>(std::clock()) * 1000000.0 /
                               static_cast<double>(CLOCKS_PER_SEC));
}

// A small, stable thread id for the trace (the real std::thread::id has no
// portable integer form).
uint64_t CurrentTid() {
  return static_cast<uint64_t>(
             std::hash<std::thread::id>{}(std::this_thread::get_id())) &
         0xffffffull;
}

double BytesToMiB(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// Minimal JSON string escaping. Span names are our own identifiers, but escape
// defensively so a future name with a quote or backslash cannot corrupt output.
std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        out += c;
    }
  }
  return out;
}

// A span pushed by Begin() and completed by End().
struct Frame {
  std::string name;
  uint64_t begin_wall_us;
  uint64_t begin_cpu_us;
  size_t begin_rss;
  size_t begin_sample_idx;
  int depth;
};

// A completed span, ready to be serialized.
struct Event {
  std::string name;
  uint64_t tid;
  uint64_t ts_us;   // start, relative to the profiler's t0
  uint64_t dur_us;  // wall-clock duration
  uint64_t cpu_us;  // CPU time consumed during the span
  size_t peak_rss;
  size_t begin_rss;
  size_t end_rss;
  int depth;
};

struct Sample {
  uint64_t ts_us;
  size_t rss;
};

// The calling thread's live span stack. File-scope thread_local so the header
// stays free of <vector>.
thread_local std::vector<Frame> g_stack;

}  // namespace

struct Profiler::Impl {
  std::string out_path;
  std::chrono::steady_clock::time_point t0;
  unsigned sample_interval_ms = 5;

  std::mutex mu;                 // guards events and samples
  std::vector<Event> events;
  std::vector<Sample> samples;

  std::thread sampler;
  std::atomic<bool> stop_sampler{false};

  uint64_t NowUs() const {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
  }
};

Profiler& Profiler::Instance() {
  static Profiler instance;
  return instance;
}

Profiler::~Profiler() { delete impl_; }

void Profiler::set_sample_interval_ms(unsigned ms) {
  if (impl_ != nullptr && ms > 0) {
    impl_->sample_interval_ms = ms;
  }
}

void Profiler::Enable(const std::string& out_path) {
  // A second Enable() (e.g. the large-model fallback simplifies twice) starts a
  // fresh run: stop any running sampler and drop prior state first.
  if (enabled_) {
    Finish();
  }
  delete impl_;
  impl_ = new Impl();
  impl_->out_path = out_path;
  impl_->t0 = std::chrono::steady_clock::now();
  g_stack.clear();

  if (const char* iv = std::getenv("ONNXSIM_PROFILE_INTERVAL_MS")) {
    const int ms = std::atoi(iv);
    if (ms > 0) {
      impl_->sample_interval_ms = static_cast<unsigned>(ms);
    }
  }

  enabled_ = true;

  // The sampler only reads process memory and appends to a mutex-guarded buffer;
  // it never touches Python, so it is safe to run while the GIL is held.
  //
  // Skipped under Emscripten, whose default build has no pthread runtime: there
  // peak memory falls back to the synchronous reads Begin()/End() take at each
  // span boundary (Finish()'s joinable() check keeps the teardown correct when
  // no thread was started).
  impl_->stop_sampler.store(false);
#if !defined(__EMSCRIPTEN__)
  Impl* impl = impl_;
  impl_->sampler = std::thread([impl]() {
    while (!impl->stop_sampler.load(std::memory_order_relaxed)) {
      const size_t rss = ReadCurrentRssBytes();
      {
        std::lock_guard<std::mutex> lk(impl->mu);
        impl->samples.push_back({impl->NowUs(), rss});
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(impl->sample_interval_ms));
    }
  });
#endif
}

void Profiler::Begin(const std::string& name) {
  if (!enabled_) {
    return;
  }
  Frame fr;
  fr.name = name;
  fr.begin_wall_us = impl_->NowUs();
  fr.begin_cpu_us = CpuMicros();
  fr.begin_rss = ReadCurrentRssBytes();
  fr.depth = static_cast<int>(g_stack.size());
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    fr.begin_sample_idx = impl_->samples.size();
  }
  g_stack.push_back(std::move(fr));
}

void Profiler::End() {
  if (!enabled_ || g_stack.empty()) {
    return;
  }
  Frame fr = std::move(g_stack.back());
  g_stack.pop_back();

  const uint64_t end_wall = impl_->NowUs();
  const uint64_t end_cpu = CpuMicros();
  const size_t end_rss = ReadCurrentRssBytes();

  Event ev;
  ev.name = std::move(fr.name);
  ev.tid = CurrentTid();
  ev.ts_us = fr.begin_wall_us;
  ev.dur_us = end_wall >= fr.begin_wall_us ? end_wall - fr.begin_wall_us : 0;
  ev.cpu_us = end_cpu >= fr.begin_cpu_us ? end_cpu - fr.begin_cpu_us : 0;
  ev.begin_rss = fr.begin_rss;
  ev.end_rss = end_rss;
  ev.depth = fr.depth;

  size_t peak = std::max(fr.begin_rss, end_rss);
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    // Samples appended after this span began fall within its window; the max of
    // those plus the synchronous begin/end reads is the span's peak RSS.
    for (size_t i = fr.begin_sample_idx; i < impl_->samples.size(); ++i) {
      peak = std::max(peak, impl_->samples[i].rss);
    }
    ev.peak_rss = peak;
    impl_->events.push_back(std::move(ev));
  }
}

namespace {

// Append one Chrome "complete" (ph:X) event for a finished span.
void WriteCompleteEvent(std::ostream& os, const Event& ev) {
  const double cpu_ms = static_cast<double>(ev.cpu_us) / 1000.0;
  const double util =
      ev.dur_us > 0 ? static_cast<double>(ev.cpu_us) / static_cast<double>(ev.dur_us)
                    : 0.0;
  os << "{\"name\":\"" << JsonEscape(ev.name) << "\",\"cat\":\"fixed_point\","
     << "\"ph\":\"X\",\"ts\":" << ev.ts_us << ",\"dur\":" << ev.dur_us
     << ",\"pid\":1,\"tid\":" << ev.tid << ",\"args\":{"
     << "\"peak_rss_mb\":" << BytesToMiB(ev.peak_rss)
     << ",\"begin_rss_mb\":" << BytesToMiB(ev.begin_rss)
     << ",\"end_rss_mb\":" << BytesToMiB(ev.end_rss) << ",\"delta_rss_mb\":"
     << (BytesToMiB(ev.end_rss) - BytesToMiB(ev.begin_rss))
     << ",\"cpu_ms\":" << cpu_ms << ",\"cpu_utilization\":" << util
     << ",\"depth\":" << ev.depth << "}}";
}

// Aggregate one function's spans for the console summary.
struct Agg {
  std::string name;
  size_t calls = 0;
  uint64_t wall_us = 0;
  uint64_t max_wall_us = 0;
  uint64_t cpu_us = 0;
  size_t peak_rss = 0;
  int min_depth = 1 << 30;
};

void PrintSummary(std::ostream& os, const std::vector<Event>& events) {
  std::unordered_map<std::string, Agg> by_name;
  for (const auto& ev : events) {
    Agg& a = by_name[ev.name];
    a.name = ev.name;
    a.calls += 1;
    a.wall_us += ev.dur_us;
    a.max_wall_us = std::max(a.max_wall_us, ev.dur_us);
    a.cpu_us += ev.cpu_us;
    a.peak_rss = std::max(a.peak_rss, ev.peak_rss);
    a.min_depth = std::min(a.min_depth, ev.depth);
  }
  std::vector<Agg> rows;
  rows.reserve(by_name.size());
  for (auto& kv : by_name) {
    rows.push_back(std::move(kv.second));
  }
  // Outermost functions first, then by total wall time descending.
  std::sort(rows.begin(), rows.end(), [](const Agg& x, const Agg& y) {
    if (x.min_depth != y.min_depth) {
      return x.min_depth < y.min_depth;
    }
    return x.wall_us > y.wall_us;
  });

  auto ms = [](uint64_t us) { return static_cast<double>(us) / 1000.0; };

  char buf[256];
  os << "\nonnxsim profiling summary (per fixed-point function)\n";
  os << "-------------------------------------------------------------------"
        "------------------\n";
  std::snprintf(buf, sizeof(buf), "%-22s %6s %12s %12s %12s %12s\n", "function",
                "calls", "wall(ms)", "cpu(ms)", "max wall(ms)", "peak(MiB)");
  os << buf;
  os << "-------------------------------------------------------------------"
        "------------------\n";
  for (const auto& a : rows) {
    // Indent by nesting depth so the hierarchy is visible in the text summary.
    std::string label(static_cast<size_t>(std::max(0, a.min_depth)) * 2, ' ');
    label += a.name;
    if (label.size() > 22) {
      label = label.substr(0, 22);
    }
    std::snprintf(buf, sizeof(buf), "%-22s %6zu %12.2f %12.2f %12.2f %12.2f\n",
                  label.c_str(), a.calls, ms(a.wall_us), ms(a.cpu_us),
                  ms(a.max_wall_us), BytesToMiB(a.peak_rss));
    os << buf;
  }
  os << "-------------------------------------------------------------------"
        "------------------\n";
}

}  // namespace

void Profiler::Finish() {
  if (!enabled_) {
    return;
  }
  enabled_ = false;

  // Stop the sampler before touching the shared buffers without the lock.
  impl_->stop_sampler.store(true, std::memory_order_relaxed);
  if (impl_->sampler.joinable()) {
    impl_->sampler.join();
  }

  // Chrome traces render cleanest with events in start order.
  std::sort(impl_->events.begin(), impl_->events.end(),
            [](const Event& a, const Event& b) { return a.ts_us < b.ts_us; });

  std::ostringstream json;
  json.setf(std::ios::fixed);
  json.precision(3);
  json << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n";

  bool first = true;
  auto comma = [&]() {
    if (!first) {
      json << ",\n";
    }
    first = false;
  };

  // Metadata: name the process and the memory-counter track.
  comma();
  json << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,"
          "\"args\":{\"name\":\"onnxsim simplify\"}}";
  comma();
  json << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,"
          "\"args\":{\"name\":\"RSS (MiB)\"}}";

  // Name each worker track after the fact so the flame graph labels threads.
  std::unordered_map<uint64_t, bool> seen_tid;
  for (const auto& ev : impl_->events) {
    if (!seen_tid[ev.tid]) {
      seen_tid[ev.tid] = true;
      comma();
      json << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":"
           << ev.tid << ",\"args\":{\"name\":\"simplify\"}}";
    }
  }

  // The span flame graph.
  for (const auto& ev : impl_->events) {
    comma();
    WriteCompleteEvent(json, ev);
  }

  // The RSS-over-time curve as counter events (a track in the timeline).
  for (const auto& s : impl_->samples) {
    comma();
    json << "{\"name\":\"RSS\",\"ph\":\"C\",\"ts\":" << s.ts_us
         << ",\"pid\":1,\"tid\":0,\"args\":{\"rss_mb\":" << BytesToMiB(s.rss)
         << "}}";
  }

  json << "\n]}\n";

  bool written = false;
  {
    std::ofstream ofs(impl_->out_path, std::ios::binary);
    if (ofs) {
      ofs << json.str();
      written = ofs.good();
    }
  }

  // Human-readable summary + where to look next.
  PrintSummary(std::cout, impl_->events);
  if (written) {
    std::cout << "onnxsim: wrote profiling trace to " << impl_->out_path
              << " (open it in chrome://tracing or https://ui.perfetto.dev to "
                 "view the flame graph)\n";
  } else {
    std::cout << "onnxsim: WARNING failed to write profiling trace to "
              << impl_->out_path << "\n";
  }
  std::cout.flush();

  delete impl_;
  impl_ = nullptr;
}

}  // namespace onnxsim
