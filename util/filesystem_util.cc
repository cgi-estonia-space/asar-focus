/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "util/filesystem_util.h"

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
}