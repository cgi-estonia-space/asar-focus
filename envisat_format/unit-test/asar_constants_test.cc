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
#include "asar_constants.h"

#include <stdexcept>

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;

TEST(AsarSpecification, GetProductTypeFromReturnsCorrectly) {
    EXPECT_THAT(
        alus::asar::specification::GetProductTypeFrom("SAR_IM__0PWDSI19920904_101949_00000018C087_00294_05950_0000.E1"),
        Eq(alus::asar::specification::ProductTypes::SAR_IM0));
    EXPECT_THAT(
        alus::asar::specification::GetProductTypeFrom("ASA_IM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1"),
        Eq(alus::asar::specification::ProductTypes::ASA_IM0));
}

TEST(AsarSpecification, GetProductTypeFromReturnsUnidentifiedForNotSupported) {
    EXPECT_THAT(
        alus::asar::specification::GetProductTypeFrom("ASA_AM__0PNPDK20040109_194924_000000182023_00157_09730_1479.N1"),
        Eq(alus::asar::specification::ProductTypes::UNIDENTIFIED));
}

TEST(AsarSpecification, TryDetermineTargetProductFromReturnsCorrectly) {
    EXPECT_THAT(alus::asar::specification::TryDetermineTargetProductFrom(
                    alus::asar::specification::ProductTypes::SAR_IM0, "IMS"),
                Eq(alus::asar::specification::ProductTypes::SAR_IMS));
    EXPECT_THAT(alus::asar::specification::TryDetermineTargetProductFrom(
                    alus::asar::specification::ProductTypes::ASA_IM0, "IMS"),
                Eq(alus::asar::specification::ProductTypes::ASA_IMS));
}

TEST(AsarSpecification, GetInstrumentFromReturnsCorrectly) {
    EXPECT_THAT(alus::asar::specification::GetInstrumentFrom(alus::asar::specification::ProductTypes::SAR_IM0),
                Eq(alus::asar::specification::Instrument::SAR));
    EXPECT_THAT(alus::asar::specification::GetInstrumentFrom(alus::asar::specification::ProductTypes::SAR_IMS),
                Eq(alus::asar::specification::Instrument::SAR));
    EXPECT_THAT(alus::asar::specification::GetInstrumentFrom(alus::asar::specification::ProductTypes::ASA_IM0),
                Eq(alus::asar::specification::Instrument::ASAR));
    EXPECT_THAT(alus::asar::specification::GetInstrumentFrom(alus::asar::specification::ProductTypes::ASA_IMS),
                Eq(alus::asar::specification::Instrument::ASAR));
    EXPECT_THROW(alus::asar::specification::GetInstrumentFrom(alus::asar::specification::ProductTypes::SAR_IMP),
                 std::runtime_error);
    EXPECT_THROW(alus::asar::specification::GetInstrumentFrom(alus::asar::specification::ProductTypes::UNIDENTIFIED),
                 std::runtime_error);
}

}  // namespace