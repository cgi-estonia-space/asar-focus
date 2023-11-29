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

namespace alus::cuda::algorithm {

template <typename T>
__device__ __host__ inline const T clamp(const T& v, const T& lo, const T& hi) {
    return v > hi ? hi : (v < lo ? lo : v);
}

}  // namespace alus::cuda::algorithm