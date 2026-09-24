// OpenCL entry points resolved from the vendor's libOpenCL.so at run time (no link-time dependency, no
// root: /vendor/lib64/libOpenCL.so is a public vendor library on Snapdragon phones).
#pragma once
#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <dlfcn.h>

#include <stdexcept>
#include <string>

#define CLFN(name) static decltype(&::name) p_##name;
#define CLFNS(X)                                                                                            \
  X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) X(clCreateCommandQueue)       \
  X(clCreateBuffer) X(clCreateProgramWithSource) X(clBuildProgram) X(clGetProgramBuildInfo)                 \
  X(clCreateKernel) X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clEnqueueWriteBuffer)                     \
  X(clEnqueueReadBuffer) X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clFinish)                       \
  X(clGetEventProfilingInfo) X(clReleaseEvent) X(clWaitForEvents) X(clGetKernelWorkGroupInfo) X(clCreateImage) X(clEnqueueWriteImage)
CLFNS(CLFN)

static void load_cl() {
  const char* paths[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so"};
  void* h = nullptr;
  for (auto p : paths)
    if ((h = dlopen(p, RTLD_NOW | RTLD_LOCAL))) break;
  if (!h) throw std::runtime_error(std::string("dlopen libOpenCL.so: ") + dlerror());
#define CLLOAD(name)                                                    \
  p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(h, #name));     \
  if (!p_##name) throw std::runtime_error("dlsym " #name);
  CLFNS(CLLOAD)
}

#define CK(x)                                                                                 \
  do {                                                                                        \
    cl_int e_ = (x);                                                                          \
    if (e_ != CL_SUCCESS) throw std::runtime_error(std::string(#x) + " = " + std::to_string(e_)); \
  } while (0)

