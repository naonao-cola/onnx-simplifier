/* Thin TVM PackedFunc ABI shim for the maxpool coverage kernel (hex_maxpool_kernel.py) -- same
 * pattern as wrapper_template.c/add_wrapper_template.c, but a single-buffer-in signature (no
 * weight/second-operand tensor -- maxpool has nothing analogous to a GEMM's B or add's second
 * operand) over uint8 packed-NCHWc data. */
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

extern void KERNEL_NAME(unsigned char* out, unsigned char* a);

int WRAPPER_NAME(TVMValue* args, int* type_codes, int num_args,
                  TVMValue* ret_val, int* ret_type_code, void* resource_handle) {
  DLTensor* out = (DLTensor*)args[0].v_handle;
  DLTensor* a = (DLTensor*)args[1].v_handle;
  KERNEL_NAME((unsigned char*)out->data, (unsigned char*)a->data);
  return 0;
}
