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
#pragma once

namespace alus::cuda::algorithm {

template <typename T>
__device__ __host__ inline const T clamp(const T& v, const T& lo, const T& hi) {
    return v > hi ? hi : (v < lo ? lo : v);
}

template <typename T>
__global__ void FillKernel(T* array, int item_count, T value) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < item_count) {
        array[x] = value;
    }
}

}  // namespace alus::cuda::algorithm