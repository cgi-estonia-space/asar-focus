/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */
#include "envisat_types.h"

#include <stdexcept>

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;

TEST(EnvisatTypesMjd, EqualsOperator) {
    mjd a{10, 10, 10};
    mjd b{10, 10, 11};
    EXPECT_THAT(a == b, IsFalse());
    b.micros = 10;
    EXPECT_THAT(a == b, IsTrue());
    EXPECT_THAT(a == a, IsTrue());
}

TEST(EnvisatTypesMjd, NotEqualsOperator) {
    mjd a{10, 10, 10};
    mjd b{10, 10, 11};
    EXPECT_THAT(a != b, IsTrue());
    b.micros = 10;
    EXPECT_THAT(a != b, IsFalse());
    EXPECT_THAT(a != a, IsFalse());
}

TEST(EnvisatTypesMjd, EqualsAreNotLessOrGreater) {
    mjd a{10, 10, 10};
    mjd b{10, 10, 10};
    EXPECT_THAT(a < b, IsFalse());
    EXPECT_THAT(b < a, IsFalse());
    EXPECT_THAT(a < a, IsFalse());
    EXPECT_THAT(a > b, IsFalse());
    EXPECT_THAT(b > a, IsFalse());
    EXPECT_THAT(a > a, IsFalse());
    EXPECT_THAT(a == b, IsTrue());
}

TEST(EnvisatTypesMjd, LessThanCorrect) {
    {
        mjd a{10, 10, 9};
        mjd b{10, 10, 10};
        EXPECT_THAT(a < b, IsTrue());
        EXPECT_THAT(b < a, IsFalse());
    }

    {
        mjd a{11, 10, 10};
        mjd b{10, 10, 10};
        EXPECT_THAT(a < b, IsFalse());
        EXPECT_THAT(b < a, IsTrue());
    }

    {
        mjd a{10, 0, 10};
        mjd b{10, 10, 10};
        EXPECT_THAT(a < b, IsTrue());
        EXPECT_THAT(b < a, IsFalse());
    }
}

TEST(EnvisatTypesMjd, GreaterThanCorrect) {
    {
        mjd a{10, 10, 9};
        mjd b{10, 10, 10};
        EXPECT_THAT(a > b, IsFalse());
        EXPECT_THAT(b > a, IsTrue());
    }

    {
        mjd a{11, 10, 10};
        mjd b{10, 10, 10};
        EXPECT_THAT(a > b, IsTrue());
        EXPECT_THAT(b > a, IsFalse());
    }

    {
        mjd a{10, 0, 10};
        mjd b{10, 10, 10};
        EXPECT_THAT(a > b, IsFalse());
        EXPECT_THAT(b > a, IsTrue());
    }
}

}  // namespace
