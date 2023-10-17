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

    bool operator <(const mjd& o) {
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

    bool operator >(const mjd& o) {
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
    uc blank[12];  // mjd?
    us isp_length;
    us crcErrorCnt;
    us correctionCnt;
    us spare;
};

static_assert(sizeof(FEPAnnotations) == 20);
