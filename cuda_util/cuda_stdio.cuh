/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#pragma once

namespace alus::cuda::stdio {

template <typename T>
__global__ void MemsetKernel(T* array, T value, int item_count) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < item_count) {
       array[x] = value;
    }
}

__global__ void MemsetKernel(void* array, void* value, int value_size_bytes, int item_count) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < item_count * value_size_bytes) {
       auto arr = (uint8_t*)array;
       auto val = (uint8_t*)value;
       const auto val_byte_chunk_index = x % value_size_bytes;
       arr[x]= val[val_byte_chunk_index];
    }
}

}