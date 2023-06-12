#pragma once

#include <cstdint>
#include <cstring>
#include <utility>

template<class T>
void bswap(T in) = delete;

inline uint32_t bswap(uint32_t in) {
    return __builtin_bswap32(in);
}

inline int32_t bswap(int32_t in) {
    uint32_t tmp;
    memcpy(&tmp, &in, 4);
    tmp = __builtin_bswap32(tmp);
    memcpy(&in, &tmp, 4);
    return in;
}

inline float bswap(float in) {
    uint32_t tmp;
    memcpy(&tmp, &in, 4);
    tmp = __builtin_bswap32(tmp);
    memcpy(&in, &tmp, 4);
    return in;
}

inline uint16_t bswap(uint16_t in) {
    return __builtin_bswap16(in);
}

inline int16_t bswap(int16_t in) {
    uint16_t tmp;
    memcpy(&tmp, &in, 2);
    tmp = __builtin_bswap16(in);
    memcpy(&in, &tmp, 2);
    return in;
}



#define BSWAP_ARR(x)  { static_assert(std::is_array_v<decltype(x)>); for(size_t ugly_macro_loop_index = 0; ugly_macro_loop_index < sizeof(x)/sizeof(x[0]); ugly_macro_loop_index++) { x[ugly_macro_loop_index] = bswap(x[ugly_macro_loop_index]); } }





