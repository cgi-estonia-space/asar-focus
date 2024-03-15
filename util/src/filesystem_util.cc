/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "filesystem_util.h"

#include <fstream>

namespace alus::util::filesystem {

std::vector<std::filesystem::path> GetFileListingRecursively(std::filesystem::path path) {
    std::vector<std::filesystem::path> listings;
    for (const auto& p : std::filesystem::recursive_directory_iterator(path)) {
        if (p.is_regular_file()) {
            listings.push_back(p);
        }
    }

    return listings;
}

std::vector<std::filesystem::path> GetFileListingAt(std::filesystem::path path, const std::regex& e) {
    std::vector<std::filesystem::path> listings;
    for (const auto& p : std::filesystem::directory_iterator(path)) {
        if (!p.is_regular_file()) {
            continue;
        }
        if (std::regex_match(p.path().filename().string(), e)) {
            listings.push_back(p);
        }
    }

    return listings;
}

std::vector<char> ReadAsBuffer(const std::filesystem::path& loc) {

    std::ifstream file(loc, std::ios::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // Read the file into a vector
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return buffer;
}
}  // namespace alus::util::filesystem