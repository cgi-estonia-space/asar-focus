/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "cuda_stdio.cuh"
#include "cuda_stdio.h"

namespace alus::cuda::stdio {

void Memset(void* d_array, void* value, size_t value_byte_size, size_t array_item_count) {
    const auto x_size = array_item_count * value_byte_size;
    dim3 block_sz(16);
    dim3 grid_sz((x_size + 15) / 16);
    MemsetKernel<<<grid_sz, block_sz>>>(d_array, value, value_byte_size, array_item_count);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

}  // namespace alus::cuda::stdio
