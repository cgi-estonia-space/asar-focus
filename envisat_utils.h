#pragma once

#include <string>

#include "checks.h"

using sc = int8_t;
using uc = uint8_t;
using ss = int16_t;
using us = uint16_t;
using sl = int32_t;
using ul = uint32_t;
using sd = int64_t;
using ud = uint64_t;
using fl = float;
using do_ = double; // do is already a keyword...




template<size_t N>
void CopyStrPad(uc (&arr)[N], const std::string &s) {
    CHECK_BOOL(N >= s.size());
    memcpy(&arr[0], s.data(), s.size());

    size_t n_pad = N - s.size();
    if (n_pad) {
        memset(&arr[s.size()], ' ', n_pad);
    }
}

