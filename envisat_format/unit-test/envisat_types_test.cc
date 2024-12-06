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
