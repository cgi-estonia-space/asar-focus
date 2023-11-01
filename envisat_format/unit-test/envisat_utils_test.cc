/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#include "envisat_utils.h"

#include <stdexcept>

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;

TEST(EnvisatUtils, MjdAddMicrosCorrect) {
    {
        mjd a{10, 10, 10};
        const auto res = MjdAddMicros(a, 180);
        EXPECT_THAT(res.days, Eq(10));
        EXPECT_THAT(res.seconds, Eq(10));
        EXPECT_THAT(res.micros, Eq(190));
    }
    {
        mjd a {1234, 42, 1000};
        const auto res = MjdAddMicros(a, 1e6);
        EXPECT_THAT(res.days, Eq(1234));
        EXPECT_THAT(res.seconds, Eq(43));
        EXPECT_THAT(res.micros, Eq(1000));
    }
}

}