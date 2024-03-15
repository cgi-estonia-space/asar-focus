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

#include <filesystem>
#include <regex>
#include <vector>

namespace alus::util::filesystem {
    std::vector<std::filesystem::path> GetFileListingRecursively(std::filesystem::path path);
    std::vector<std::filesystem::path> GetFileListingAt(std::filesystem::path path, const std::regex& e);
    std::vector<char> ReadAsBuffer(const std::filesystem::path& loc);
}