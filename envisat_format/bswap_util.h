#pragma once

#include <climits>
#include <cstdint>
#include <cstring>
#include <utility>

#include "envisat_types.h"

// Much of the byteparsing and writing code in this project written with the following assumptions
static_assert(CHAR_BIT == 8);
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
// if either of these asserts fail, you will want to double check all memcpyes, bswap() and member BSwap() calls

template <class T>
void bswap(T in) = delete;  // avoid implicit conversions

[[nodiscard]] inline uint32_t bswap(uint32_t in) { return __builtin_bswap32(in); }

[[nodiscard]] inline int32_t bswap(int32_t in) {
    uint32_t tmp;
    memcpy(&tmp, &in, 4);
    tmp = __builtin_bswap32(tmp);
    memcpy(&in, &tmp, 4);
    return in;
}

[[nodiscard]] inline float bswap(float in) {
    uint32_t tmp;
    memcpy(&tmp, &in, 4);
    tmp = __builtin_bswap32(tmp);
    memcpy(&in, &tmp, 4);
    return in;
}

[[nodiscard]] inline uint16_t bswap(uint16_t in) { return __builtin_bswap16(in); }

[[nodiscard]] inline int16_t bswap(int16_t in) {
    uint16_t tmp;
    memcpy(&tmp, &in, 2);
    tmp = __builtin_bswap16(in);
    memcpy(&in, &tmp, 2);
    return in;
}

// Unfortunately has to be a macro, because often used for byteswapping members of packed structs
#define BSWAP_ARR(x)                                                                             \
    {                                                                                            \
        static_assert(std::is_array_v<decltype(x)>);                                             \
        for (size_t ugly_macro_loop_index = 0; ugly_macro_loop_index < sizeof(x) / sizeof(x[0]); \
             ugly_macro_loop_index++) {                                                          \
            x[ugly_macro_loop_index] = bswap(x[ugly_macro_loop_index]);                          \
        }                                                                                        \
    }

[[nodiscard]] inline mjd bswap(mjd in) {
    in.days = bswap(in.days);
    in.seconds = bswap(in.seconds);
    in.micros = bswap(in.micros);
    return in;
}

[[nodiscard]] inline FEPAnnotations bswap(FEPAnnotations in) {
    in.isp_length = bswap(in.isp_length);
    in.crcErrorCnt = bswap(in.crcErrorCnt);
    in.correctionCnt = bswap(in.correctionCnt);
    return in;
}
