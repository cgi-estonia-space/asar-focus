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
