/* Thin TVM PackedFunc ABI shim for the elementwise-add coverage kernel (hex_add_kernel.py) --
 * same pattern as wrapper_template.c, but for an int32+int32->int32 signature (the real
 * backbone's `add` op operates on int32 pre-requantization accumulators, not uint8, unlike the
 * GEMM kernels' wrapper_template.c) instead of touching that shared file. */
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

extern void KERNEL_NAME(int* out, int* a, int* b);

int WRAPPER_NAME(TVMValue* args, int* type_codes, int num_args,
                  TVMValue* ret_val, int* ret_type_code, void* resource_handle) {
  DLTensor* out = (DLTensor*)args[0].v_handle;
  DLTensor* a = (DLTensor*)args[1].v_handle;
  DLTensor* b = (DLTensor*)args[2].v_handle;
  KERNEL_NAME((int*)out->data, (int*)a->data, (int*)b->data);
  return 0;
}
