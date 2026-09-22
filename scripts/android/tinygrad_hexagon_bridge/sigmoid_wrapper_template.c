/* Thin TVM PackedFunc ABI shim for the sigmoid coverage kernel -- same pattern as
 * wrapper_template.c/add_wrapper_template.c/maxpool_wrapper_template.c, but a single-buffer-in,
 * float32 signature (elementwise, no accumulator, no second operand). */
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

extern void KERNEL_NAME(float* out, float* a);

int WRAPPER_NAME(TVMValue* args, int* type_codes, int num_args,
                  TVMValue* ret_val, int* ret_type_code, void* resource_handle) {
  DLTensor* out = (DLTensor*)args[0].v_handle;
  DLTensor* a = (DLTensor*)args[1].v_handle;
  KERNEL_NAME((float*)out->data, (float*)a->data);
  return 0;
}
