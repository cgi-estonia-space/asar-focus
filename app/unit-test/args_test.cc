/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "args.h"

#include <vector>

#include "gmock/gmock.h"

namespace {

const char* DEFAULT_ARGV_OK[] = {"-i",    "/tmp/ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1",
                                 "--aux", "/tmp/ASAR_Auxiliary_Files",
                                 "--orb", "/tmp/DOR_VOR_AXVF-P20120424_120600_20040108_215528_20040110_002328",
                                 "-o",    "/tmp",
                                 "-t",    "IMS"};
const int DEFAULT_ARGV_OK_COUNT{sizeof(DEFAULT_ARGV_OK)};
const std::vector<char*> DEFAULT_ARGV_OK_VECTOR{(char*)"program",  // Boost parse_command_line() skips first item.
                                                const_cast<char*>(DEFAULT_ARGV_OK[0]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[1]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[2]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[3]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[4]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[5]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[6]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[7]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[8]),
                                                const_cast<char*>(DEFAULT_ARGV_OK[9])};

using ::testing::Eq;
using ::testing::IsTrue;
using ::testing::StrEq;

TEST(ArgsTest, CompleteArgumentsAreParsedCorrectly) {
    alus::asar::Args a(DEFAULT_ARGV_OK_VECTOR);
    EXPECT_THAT(std::string(a.GetInputDsPath()), StrEq(DEFAULT_ARGV_OK[1]));
    EXPECT_THAT(std::string(a.GetAuxPath()), StrEq(DEFAULT_ARGV_OK[3]));
    EXPECT_THAT(std::string(a.GetOrbitPath()), StrEq(DEFAULT_ARGV_OK[5]));
    EXPECT_THAT(std::string(a.GetOutputPath()), StrEq(DEFAULT_ARGV_OK[7]));
    EXPECT_THAT(std::string(a.GetFocussedProductType()), StrEq(DEFAULT_ARGV_OK[9]));
}

TEST(ArgsTest, IsHelpRequestedScenarios) {
    {
        const char* argv[] = {"--help"};
        std::vector<char*> ta{const_cast<char*>(argv[0])};
        alus::asar::Args a(ta);
        EXPECT_THAT(a.IsHelpRequested(), IsTrue());
    }
    {
        std::vector<char*> ta{};
        alus::asar::Args a(ta);
        EXPECT_THAT(a.IsHelpRequested(), IsTrue());
    }
    {
        const char* argv[] = {"-h"};
        std::vector<char*> ta{const_cast<char*>(argv[0])};
        alus::asar::Args a(ta);
        EXPECT_THAT(a.IsHelpRequested(), IsTrue());
    }
}

}  // namespace