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

#include <cstddef>
#include <unistd.h>

namespace alus::util {

// Refer to "alus-experiments" host_memory_allocation.cc for speed tests.
inline void* Memalloc(size_t bytes) {
    char* buf = new char[bytes];
    for (size_t pg{0}; pg < bytes / sysconf(_SC_PAGE_SIZE); pg++) {
        buf[pg * 4096] = 0x00;
    }
    return buf;
}

}
