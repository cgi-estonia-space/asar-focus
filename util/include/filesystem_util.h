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

#include <filesystem>
#include <regex>
#include <vector>

namespace alus::util::filesystem {
    std::vector<std::filesystem::path> GetFileListingRecursively(std::filesystem::path path);
    std::vector<std::filesystem::path> GetFileListingAt(std::filesystem::path path, const std::regex& e);
    std::vector<char> ReadAsBuffer(const std::filesystem::path& loc);
}