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



#include "patc.h"

#include <cstring>

#include "filesystem_util.h"

namespace {

alus::asar::envformat::aux::Patc ConstructPatc(const std::vector<char>& buffer) {
    alus::asar::envformat::aux::Patc patc;

    patc.cyclic_counter = std::stoul(std::string(buffer.data() + 15, 4));
    patc.mission[0] = buffer.at(20);
    patc.mission[1] = buffer.at(21);
    patc.orbit_number = std::stoul(std::string(buffer.data() + 30, 5));
    memcpy(&patc.epoch_1950, buffer.data() + 38, 4);
    memcpy(&patc.milliseconds, buffer.data() + 42, 4);

    memcpy(&patc.sbt_counter, buffer.data() + 46, 4);
    memcpy(&patc.sbt_period, buffer.data() + 50, 4);

    return patc;
}

}  // namespace

namespace alus::asar::envformat::aux {

Patc ParsePatc(const std::filesystem::path& file_loc) {
    const auto& patc_buffer = util::filesystem::ReadAsBuffer(file_loc);

    if (patc_buffer.size() != 54) {
        throw std::runtime_error("PATC file '" + file_loc.string() + "' should be 54 bytes, actual - " +
                                 std::to_string(patc_buffer.size()));
    }

    return ConstructPatc(patc_buffer);
}

}  // namespace alus::asar::envformat::aux