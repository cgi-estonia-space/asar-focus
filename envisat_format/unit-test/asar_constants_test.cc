/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
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

}  // namespace