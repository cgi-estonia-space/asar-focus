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
#include <filesystem>

namespace alus::asar::envformat::aux {

// Should not be under this module. An exception, as a surprise, there is a file format
// which is NOT packaged into envisat format, nice.
struct Patc {
    uint16_t cyclic_counter;
    char mission[2];
    uint32_t orbit_number;
    int32_t epoch_1950; // Days
    int32_t milliseconds;
    uint32_t sbt_counter;
    uint32_t sbt_period;  // Tick time value in nanoseconds
};

Patc ParsePatc(const std::filesystem::path& file_loc);

}  // namespace alus::asar::envformat::aux