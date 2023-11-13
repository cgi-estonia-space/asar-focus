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

#include <algorithm>

namespace alus::cuda::bit {

__device__ inline float byteswap(const float& v) {
    uint32_t int_value;
    std::memcpy(&int_value, &v, sizeof(float));

    uint32_t swapped_value = ((int_value >> 24) & 0xFF) | ((int_value >> 8) & 0xFF00) | ((int_value << 8) & 0xFF0000) |
                             ((int_value << 24) & 0xFF000000);

    float result;
    std::memcpy(&result, &swapped_value, sizeof(float));

    return result;
}

}  // namespace alus::cuda::bit
