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
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::StrEq;

TEST(ArgsTest, CompleteArgumentsAreParsedCorrectly) {
    {
        alus::asar::Args a(DEFAULT_ARGV_OK_VECTOR);
        EXPECT_THAT(std::string(a.GetInputDsPath()), StrEq(DEFAULT_ARGV_OK[1]));
        EXPECT_THAT(std::string(a.GetAuxPath()), StrEq(DEFAULT_ARGV_OK[3]));
        EXPECT_THAT(std::string(a.GetOrbitPath()), StrEq(DEFAULT_ARGV_OK[5]));
        EXPECT_THAT(std::string(a.GetOutputPath()), StrEq(DEFAULT_ARGV_OK[7]));
        EXPECT_THAT(std::string(a.GetFocussedProductType()), StrEq(DEFAULT_ARGV_OK[9]));
        EXPECT_THAT(a.GetProcessSensingStart(), Eq(std::nullopt));
        EXPECT_THAT(a.GetProcessSensingEnd(), Eq(std::nullopt));
    }
    {
        const char* log_arg[] = {"--log", "iNfO"};
        const std::vector<char*> arg_vector{(char*)"program",  // Boost parse_command_line() skips first item.
                                            const_cast<char*>(DEFAULT_ARGV_OK[0]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[1]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[2]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[3]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[4]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[5]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[6]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[7]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[8]),
                                            const_cast<char*>(DEFAULT_ARGV_OK[9]),
                                            const_cast<char*>(log_arg[0]),
                                            const_cast<char*>(log_arg[1])};
        alus::asar::Args a(arg_vector);
        EXPECT_THAT(a.GetLogLevel(), Eq(alus::asar::log::Level::INFO));
    }
}

TEST(ArgsTest, IsHelpRequestedScenarios) {
    {
        const char* argv[] = {"program", "--help"};
        std::vector<char*> ta{const_cast<char*>(argv[0]), const_cast<char*>(argv[1])};
        alus::asar::Args a(ta);
        EXPECT_THAT(a.IsHelpRequested(), IsTrue());
    }
    {
        const char* argv[] = {"program"};
        std::vector<char*> ta{const_cast<char*>(argv[0])};
        alus::asar::Args a(ta);
        EXPECT_THAT(a.IsHelpRequested(), IsTrue());
    }
    {
        const char* argv[] = {"program", "-h"};
        std::vector<char*> ta{const_cast<char*>(argv[0]), const_cast<char*>(argv[1])};
        alus::asar::Args a(ta);
        EXPECT_THAT(a.IsHelpRequested(), IsTrue());
    }
}

TEST(ArgsTest, ThrowsLogicErrorWhenEmptyArraySupplied) {
    ASSERT_THROW(alus::asar::Args(std::vector<char*>{}), std::logic_error);
}

TEST(ArgsTest, ThrowsInvalidArgumentErrorWhenLogLevelHasInvalidString) {
    const char* log_arg[] = {"--log", "ViNfO"};
    const std::vector<char*> arg_vector{(char*)"program",  // Boost parse_command_line() skips first item.
                                        const_cast<char*>(DEFAULT_ARGV_OK[0]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[1]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[2]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[3]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[4]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[5]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[6]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[7]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[8]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[9]),
                                        const_cast<char*>(log_arg[0]),
                                        const_cast<char*>(log_arg[1])};
    ASSERT_THROW(alus::asar::Args({arg_vector}), std::invalid_argument);
}

TEST(ArgsTest, HiddenArgsAreFalseByDefault) {
    alus::asar::Args a(DEFAULT_ARGV_OK_VECTOR);
    EXPECT_THAT(a.StorePlots(), IsFalse());
    EXPECT_THAT(a.StoreIntensity(), IsFalse());
}

TEST(ArgsTest, HiddenArgsAreDetected) {
    const char* hidden_args[] = {"--plot", "--intensity"};
    const std::vector<char*> arg_vector{(char*)"program",  // Boost parse_command_line() skips first item.
                                        const_cast<char*>(DEFAULT_ARGV_OK[0]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[1]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[2]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[3]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[4]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[5]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[6]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[7]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[8]),
                                        const_cast<char*>(DEFAULT_ARGV_OK[9]),
                                        const_cast<char*>(hidden_args[0]),
                                        const_cast<char*>(hidden_args[1])};
    alus::asar::Args a(arg_vector);
    EXPECT_THAT(a.StorePlots(), IsTrue());
    EXPECT_THAT(a.StoreIntensity(), IsTrue());
}

}  // namespace