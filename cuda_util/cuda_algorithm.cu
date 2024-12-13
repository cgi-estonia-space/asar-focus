/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
