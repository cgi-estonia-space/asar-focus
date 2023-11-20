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

#include <cstddef>
#include <unistd.h>

namespace alus::util {

// Refer to experiments/host_memory_allocation.cc for speed tests.
inline void* Memalloc(size_t bytes) {
    char* buf = new char[bytes];
    for (size_t pg{0}; pg < bytes / sysconf(_SC_PAGE_SIZE); pg++) {
        buf[pg * 4096] = 0x00;
    }
    return buf;
}

}