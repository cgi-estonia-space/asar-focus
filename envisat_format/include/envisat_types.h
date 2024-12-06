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

#include <cstdint>

// Names and sizes from
// ENVISAT-1 PRODUCTS SPECIFICATIONS
// ANNEX A: PRODUCT DATA CONVENTIONS
// IDEAS+-SER-IPF-SPE-2337

using sc = int8_t;
using uc = uint8_t;
using ss = int16_t;
using us = uint16_t;
using sl = int32_t;
using ul = uint32_t;
using sd = int64_t;
using ud = uint64_t;
using fl = float;
using do_ = double;  // do is already a keyword...

struct mjd {
    sl days;
    ul seconds;
    ul micros;

    bool operator ==(const mjd& o) const{
        return days == o.days && seconds == o.seconds && micros == o.micros;
    }

    bool operator !=(const mjd& o) const{
        return days != o.days || seconds != o.seconds || micros != o.micros;
    }

    bool operator <(const mjd& o) const{
        if (days == o.days) {
            if (seconds == o.seconds) {
                return micros < o.micros;
            } else {
                return seconds < o.seconds;
            }
        } else {
            return days < o.days;
        }
    }

    bool operator >(const mjd& o) const {
        if (days == o.days) {
            if (seconds == o.seconds) {
                return micros > o.micros;
            } else {
                return seconds > o.seconds;
            }
        } else {
            return days > o.days;
        }
    }
};

static_assert(sizeof(mjd) == 12);

struct FEPAnnotations {
    mjd mjd_time;
    us isp_length;
    us crcErrorCnt;
    us correctionCnt;
    us spare;
};

static_assert(sizeof(FEPAnnotations) == 20);

struct IQ16 {
    int16_t i;
    int16_t q;
};

static_assert(sizeof(IQ16) == 4);
