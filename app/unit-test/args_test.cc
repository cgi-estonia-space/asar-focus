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

#include "args.h"

#include <vector>

#include "gmock/gmock.h"

namespace {

template <typename T, std::size_t N>
constexpr std::size_t SizeOfArray(const T (&)[N]) {
    return N;
}

// Boost parse_command_line() skips first item.
const char* DEFAULT_ARGV_OK[] = {"program",
                                 "-i",
                                 "/tmp/ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1",
                                 "--aux",
                                 "/tmp/ASAR_Auxiliary_Files",
                                 "--orb",
                                 "/tmp/DOR_VOR_AXVF-P20120424_120600_20040108_215528_20040110_002328",
                                 "-o",
                                 "/tmp",
                                 "-t",
                                 "IMS"};
constexpr size_t DEFAULT_ARGV_OK_COUNT{SizeOfArray(DEFAULT_ARGV_OK)};

std::vector<char*> ConstructArgsVector(const char* args[], size_t length) {
    auto begin = const_cast<char**>(&args[0]);
    std::vector<char*> v(begin, begin + length);
    return v;
}

void AddArgsToVector(std::vector<char*>& args, const char* args_to_add[], size_t count) {
    for (size_t i{}; i < count; i++) {
        args.push_back(const_cast<char*>(args_to_add[i]));
    }
}

using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::StrEq;

TEST(ArgsTest, CompleteArgumentsAreParsedCorrectly) {
    {
        alus::asar::Args a(ConstructArgsVector(DEFAULT_ARGV_OK, DEFAULT_ARGV_OK_COUNT));
        EXPECT_THAT(std::string(a.GetInputDsPath()), StrEq(DEFAULT_ARGV_OK[2]));
        EXPECT_THAT(std::string(a.GetAuxPath()), StrEq(DEFAULT_ARGV_OK[4]));
        EXPECT_THAT(std::string(a.GetOrbitPath()), StrEq(DEFAULT_ARGV_OK[6]));
        EXPECT_THAT(std::string(a.GetOutputPath()), StrEq(DEFAULT_ARGV_OK[8]));
        EXPECT_THAT(std::string(a.GetFocussedProductType()), StrEq(DEFAULT_ARGV_OK[10]));
        EXPECT_THAT(a.GetProcessSensingStart(), Eq(std::nullopt));
        EXPECT_THAT(a.GetProcessSensingEnd(), Eq(std::nullopt));
    }
    {
        const char* log_arg[] = {"--log", "iNfO"};
        auto arg_vector = ConstructArgsVector(DEFAULT_ARGV_OK, DEFAULT_ARGV_OK_COUNT);
        AddArgsToVector(arg_vector, log_arg, SizeOfArray(log_arg));
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
    auto arg_vector = ConstructArgsVector(DEFAULT_ARGV_OK, DEFAULT_ARGV_OK_COUNT);
    AddArgsToVector(arg_vector, log_arg, SizeOfArray(log_arg));

    ASSERT_THROW(alus::asar::Args({arg_vector}), std::invalid_argument);
}

TEST(ArgsTest, HiddenArgsAreFalseByDefault) {
    alus::asar::Args a(ConstructArgsVector(DEFAULT_ARGV_OK, DEFAULT_ARGV_OK_COUNT));
    EXPECT_THAT(a.StorePlots(), IsFalse());
    EXPECT_THAT(a.StoreIntensity(), IsFalse());
}

TEST(ArgsTest, HiddenArgsAreDetected) {
    const char* hidden_args[] = {"--plot", "--intensity"};
    auto arg_vector = ConstructArgsVector(DEFAULT_ARGV_OK, DEFAULT_ARGV_OK_COUNT);
    AddArgsToVector(arg_vector, hidden_args, SizeOfArray(hidden_args));

    alus::asar::Args a(arg_vector);
    EXPECT_THAT(a.StorePlots(), IsTrue());
    EXPECT_THAT(a.StoreIntensity(), IsTrue());
}

}  // namespace