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

#include "cuda_util.h"

namespace alus::cuda::stdio {

// The underlying type of the 'value' shall be the same as the d_array,
// otherwise illegal memory access or some elements are not set.
// This indeed is similar to original CUDA runtime API's cudaMemset, but that one sets by byte.
// This one here enables to set structs, floats etc... more than byte arrays.
void Memset(void* d_array, void* value, size_t value_byte_size, size_t array_item_count);

template <typename T>
void Memset(T* d_array, T value, size_t count) {
    T* d_value;
    CHECK_CUDA_ERR(cudaMalloc(&d_value, sizeof(T)));
    CHECK_CUDA_ERR(cudaMemcpy(d_value, &value, sizeof(T), cudaMemcpyHostToDevice));
    Memset(d_array, d_value, sizeof(T), count);
}

}  // namespace alus::cuda::stdio