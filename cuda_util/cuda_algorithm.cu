/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "cuda_algorithm.h"

#include <cufft.h>

#include "cuda_algorithm.cuh"
#include "cuda_util.h"

namespace alus::cuda::algorithm {

template <typename T>
void Fill(T* d_array, size_t count, T value) {
    const int x_size = count;
    dim3 block_sz(16);
    dim3 grid_sz((x_size + 15) / 16);
    FillKernel<<<grid_sz, block_sz>>>(d_array, x_size, value);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

template void Fill<cufftComplex>(cufftComplex*, size_t, cufftComplex);

}  // namespace alus::cuda::algorithm
