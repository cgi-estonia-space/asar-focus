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

#include <cstdint>
#include <limits>
#include <type_traits>

namespace alus::asar::envformat::parseutil {

template <typename T, T BITS>
constexpr T MaxValueForBits() {
    constexpr uint64_t value = (1ull << BITS) - 1;
    static_assert(value <= std::numeric_limits<T>::max());
    return static_cast<T>(value);
}

template <typename T, T LIMIT>
inline T CounterGap(T last_value, T current_value) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned type");
    static_assert(LIMIT <= std::numeric_limits<T>::max(), "LIMIT must be within the range of T");
    // This aritchmetically equals the last 'else' condition.
    // It is expected that this condition is most probable first hit.
    if (last_value + 1 == current_value) {
        return 1;
    } else if (current_value < last_value) {  // Also covers when 'last_value == LIMIT'
        return LIMIT - last_value + current_value + 1;
    } else {
        return current_value - last_value;
    }
}

}  // namespace alus::asar::envformat::parseutil