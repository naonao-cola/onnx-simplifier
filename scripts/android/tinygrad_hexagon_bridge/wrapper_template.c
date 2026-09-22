/* Thin TVM PackedFunc ABI shim exposing a tinygrad-generated Hexagon kernel (captured by
 * capture_kernel.py) so it can be loaded and called through TVM's existing, already-permitted
 * Hexagon RPC transport (session.load_module + mod.get_function) instead of tinygrad's own DSP
 * driver, which needs raw /dev/adsprpc-smd access that production Android builds deny to an
 * unprivileged process (see capture_kernel.py's docstring). Only the transport is TVM's; the
 * compute kernel is tinygrad's own generated code, compiled separately with the same Hexagon
 * toolchain TVM itself uses and linked together via tvm.contrib.hexagon.tools.link_shared.
 *
 * Deliberately plain C (not C++): TVM's C++ TVM_DLL_EXPORT_TYPED_FUNC macro pulls in
 * tvm::runtime::Array/Map, which need more libc++ (pthread.h, sched.h) than this freestanding
 * Hexagon target build provides. The raw C packed-function ABI (TVMValue* / DLTensor*) needs
 * only the plain-C headers below.
 *
 * KERNEL_NAME and WRAPPER_NAME are substituted by bridge_and_test.py to match the actual
 * captured kernel's function name -- tinygrad names kernels after their shape (e.g.
 * r_128_2_2_2_2_2_2_4_4_16_2_2), so this can't be hardcoded.
 */
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

extern void KERNEL_NAME(int* out, unsigned char* a, unsigned char* b);

int WRAPPER_NAME(TVMValue* args, int* type_codes, int num_args,
                  TVMValue* ret_val, int* ret_type_code, void* resource_handle) {
  DLTensor* out = (DLTensor*)args[0].v_handle;
  DLTensor* a = (DLTensor*)args[1].v_handle;
  DLTensor* b = (DLTensor*)args[2].v_handle;
  KERNEL_NAME((int*)out->data, (unsigned char*)a->data, (unsigned char*)b->data);
  return 0;
}
