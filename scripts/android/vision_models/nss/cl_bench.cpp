// Time one OpenCL kernel on the phone's GPU: used to compare tinygrad-generated kernels (rendered by
// tinygrad's OpenCL backend on the host) with hand-written ones on the same inputs.
//
//   cl_bench <file.cl> <kernel> <gx,gy,gz> <lx,ly,lz | -> <iters> <arg>...
//     arg: in:<file>            read-only buffer from a file
//          out:<bytes>:<file>   buffer written back to <file> after the last run
//          f:<float> | i:<int>  scalar
// Prints the median kernel time (OpenCL profiling events).
//
//   cl_bench --plan <plan.txt> <iters>
//     a sequence of kernels sharing buffers (e.g. all kernels tinygrad scheduled for one graph), lines:
//       BUF <name> <bytes> [<input file>]
//       K <file.cl> <kernel> <gx,gy,gz> <lx,ly,lz|-> <arg>...   (arg: a BUF name, f:<float> or i:<int>)
//       OUT <name> <file>
//     prints the median of the per-iteration sum of kernel times, and each kernel's median.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cl_dl.h"

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("open " + p);
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

static std::vector<size_t> dims(const std::string& s) {
  std::vector<size_t> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(std::stoul(t));
  return v;
}

static int run_plan(cl_context ctx, cl_device_id dev, cl_command_queue q, const char* path, int iters);

int main(int argc, char** argv) try {
  if (argc >= 4 && std::string(argv[1]) == "--plan") {
    load_cl();
    cl_platform_id plat;
    CK(p_clGetPlatformIDs(1, &plat, nullptr));
    cl_device_id dev;
    CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    cl_int err;
    cl_context ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CK(err);
    cl_command_queue q = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    CK(err);
    return run_plan(ctx, dev, q, argv[2], atoi(argv[3]));
  }
  if (argc < 6) {
    fprintf(stderr, "usage: cl_bench <file.cl> <kernel> <gx,gy,gz> <lx,ly,lz|-> <iters> <arg>...\n");
    return 2;
  }
  load_cl();
  cl_platform_id plat;
  CK(p_clGetPlatformIDs(1, &plat, nullptr));
  cl_device_id dev;
  CK(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
  cl_int err;
  cl_context ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  CK(err);
  cl_command_queue q = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
  CK(err);
  auto src = read_file(argv[1]);
  const char* sp = src.data();
  size_t sl = src.size();
  cl_program prog = p_clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
  CK(err);
  if (p_clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
    std::vector<char> log(1 << 16);
    p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    fprintf(stderr, "%s\n", log.data());
    return 1;
  }
  cl_kernel k = p_clCreateKernel(prog, argv[2], &err);
  CK(err);
  std::vector<size_t> g = dims(argv[3]), l;
  if (std::string(argv[4]) != "-") l = dims(argv[4]);
  int iters = atoi(argv[5]);
  std::vector<std::pair<cl_mem, std::pair<size_t, std::string>>> outs;
  std::vector<std::vector<char>> keep;
  for (int i = 6; i < argc; i++) {
    std::string a = argv[i];
    cl_uint ai = i - 6;
    if (a.rfind("in:", 0) == 0) {
      keep.push_back(read_file(a.substr(3)));
      cl_mem m = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, keep.back().size(),
                                  keep.back().data(), &err);
      CK(err);
      CK(p_clSetKernelArg(k, ai, sizeof m, &m));
    } else if (a.rfind("out:", 0) == 0) {
      size_t c = a.find(':', 4);
      size_t bytes = std::stoul(a.substr(4, c - 4));
      cl_mem m = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
      CK(err);
      CK(p_clSetKernelArg(k, ai, sizeof m, &m));
      outs.push_back({m, {bytes, a.substr(c + 1)}});
    } else if (a.rfind("f:", 0) == 0) {
      float v = std::stof(a.substr(2));
      CK(p_clSetKernelArg(k, ai, sizeof v, &v));
    } else if (a.rfind("i:", 0) == 0) {
      int v = std::stoi(a.substr(2));
      CK(p_clSetKernelArg(k, ai, sizeof v, &v));
    } else {
      throw std::runtime_error("bad arg " + a);
    }
  }
  std::vector<double> ms;
  for (int it = 0; it < iters + 2; it++) {
    cl_event ev;
    CK(p_clEnqueueNDRangeKernel(q, k, (cl_uint)g.size(), nullptr, g.data(), l.empty() ? nullptr : l.data(), 0,
                                nullptr, &ev));
    CK(p_clWaitForEvents(1, &ev));
    cl_ulong a, b;
    p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr);
    p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr);
    p_clReleaseEvent(ev);
    if (it >= 2) ms.push_back((b - a) * 1e-6);
  }
  std::sort(ms.begin(), ms.end());
  printf("%s median %.3f ms (min %.3f, n=%zu)\n", argv[2], ms[ms.size() / 2], ms[0], ms.size());
  for (auto& o : outs) {
    std::vector<char> h(o.second.first);
    CK(p_clEnqueueReadBuffer(q, o.first, CL_TRUE, 0, h.size(), h.data(), 0, nullptr, nullptr));
    std::ofstream(o.second.second, std::ios::binary).write(h.data(), h.size());
  }
  return 0;
} catch (const std::exception& ex) {
  fprintf(stderr, "error: %s\n", ex.what());
  return 1;
}

static int run_plan(cl_context ctx, cl_device_id dev, cl_command_queue q, const char* path, int iters) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error(std::string("open ") + path);
  std::string line;
  std::unordered_map<std::string, cl_mem> bufs;
  std::unordered_map<std::string, size_t> sizes;
  struct Kn {
    cl_kernel k;
    std::string name;
    std::vector<size_t> g, l;
  };
  std::vector<Kn> ks;
  std::vector<std::pair<std::string, std::string>> outs;
  std::unordered_map<std::string, cl_program> progs;
  cl_int err;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "BUF") {
      std::string name, file;
      size_t bytes;
      ss >> name >> bytes >> file;
      cl_mem m;
      if (!file.empty()) {
        auto d = read_file(file);
        m = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d.size(), d.data(), &err);
      } else {
        m = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
      }
      CK(err);
      bufs[name] = m;
      sizes[name] = bytes;
    } else if (tag == "K") {
      std::string file, kname, g, l, a;
      ss >> file >> kname >> g >> l;
      if (!progs.count(file)) {
        auto src = read_file(file);
        const char* sp = src.data();
        size_t sl = src.size();
        cl_program pr = p_clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
        CK(err);
        if (p_clBuildProgram(pr, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
          std::vector<char> log(1 << 16);
          p_clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
          fprintf(stderr, "%s\n", log.data());
          return 1;
        }
        progs[file] = pr;
      }
      Kn kn{p_clCreateKernel(progs[file], kname.c_str(), &err), kname, dims(g), {}};
      CK(err);
      if (l != "-") kn.l = dims(l);
      cl_uint ai = 0;
      while (ss >> a) {
        if (a.rfind("f:", 0) == 0) {
          float v = std::stof(a.substr(2));
          CK(p_clSetKernelArg(kn.k, ai++, sizeof v, &v));
        } else if (a.rfind("i:", 0) == 0) {
          int v = std::stoi(a.substr(2));
          CK(p_clSetKernelArg(kn.k, ai++, sizeof v, &v));
        } else {
          if (!bufs.count(a)) throw std::runtime_error("unknown buffer " + a);
          CK(p_clSetKernelArg(kn.k, ai++, sizeof(cl_mem), &bufs[a]));
        }
      }
      ks.push_back(kn);
    } else if (tag == "OUT") {
      std::string name, file;
      ss >> name >> file;
      outs.push_back({name, file});
    }
  }
  std::vector<double> tot;
  std::vector<std::vector<double>> per(ks.size());
  for (int it = 0; it < iters + 2; it++) {
    double sum = 0;
    for (size_t i = 0; i < ks.size(); i++) {
      cl_event ev;
      CK(p_clEnqueueNDRangeKernel(q, ks[i].k, (cl_uint)ks[i].g.size(), nullptr, ks[i].g.data(),
                                  ks[i].l.empty() ? nullptr : ks[i].l.data(), 0, nullptr, &ev));
      CK(p_clWaitForEvents(1, &ev));
      cl_ulong a, b;
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr);
      p_clReleaseEvent(ev);
      if (it >= 2) per[i].push_back((b - a) * 1e-6);
      sum += (b - a) * 1e-6;
    }
    if (it >= 2) tot.push_back(sum);
  }
  std::sort(tot.begin(), tot.end());
  printf("%s: %zu kernel(s), median total %.3f ms\n", path, ks.size(), tot[tot.size() / 2]);
  for (size_t i = 0; i < ks.size(); i++) {
    std::sort(per[i].begin(), per[i].end());
    printf("  %s %.3f ms\n", ks[i].name.c_str(), per[i][per[i].size() / 2]);
  }
  for (auto& o : outs) {
    std::vector<char> h(sizes[o.first]);
    CK(p_clEnqueueReadBuffer(q, bufs[o.first], CL_TRUE, 0, h.size(), h.data(), 0, nullptr, nullptr));
    std::ofstream(o.second, std::ios::binary).write(h.data(), h.size());
  }
  return 0;
}
