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

#include "date_time_util.h"

#include <locale>
#include <stdexcept>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;

TEST(DateTimeUtil, ParseDateReturnsCorrectByTheFormat) {
    constexpr std::string_view DATE_PATTERN{"%Y%m%d_%H%M%S"};
    const auto* timestamp_styling = new boost::posix_time::time_input_facet(DATE_PATTERN.data());
    const auto locale = std::locale(std::locale::classic(), timestamp_styling);

    const auto date_str{"20021231 123455"};
    auto result{alus::util::date_time::ParseDate(date_str, locale)};
    EXPECT_THAT(result.date().year(), Eq(2002));
    EXPECT_THAT(result.date().month(), Eq(12));
    EXPECT_THAT(result.date().day(), Eq(31));
    EXPECT_THAT(result.time_of_day().hours(), Eq(12));
    EXPECT_THAT(result.time_of_day().minutes(), Eq(34));
    EXPECT_THAT(result.time_of_day().seconds(), Eq(55));
}

TEST(DateTimeUtil, ToYyyyMmDdHhMmSsfReturnsCorrect) {
    const auto result{alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff("20040229_212504912000")};
    EXPECT_THAT(result.date().year(), Eq(2004));
    EXPECT_THAT(result.date().month(), Eq(2));
    EXPECT_THAT(result.date().day(), Eq(29));
    EXPECT_THAT(result.time_of_day().hours(), Eq(21));
    EXPECT_THAT(result.time_of_day().minutes(), Eq(25));
    EXPECT_THAT(result.time_of_day().seconds(), Eq(4));
    EXPECT_THAT(result.time_of_day().fractional_seconds(), Eq(912000));
}

TEST(DateTimeUtil, Tere) {
    // No delimiter - '_'
    EXPECT_THROW(alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff("200402292125049120000"), std::invalid_argument);
    // Shorter
    EXPECT_THROW(alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff("20040229_21250491200"), std::invalid_argument);
    // Extra day
    EXPECT_THROW(alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff("20040230_242504912000"), std::invalid_argument);
    // Longer
    EXPECT_THROW(alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff("20040229_2125049120000"), std::invalid_argument);
}

}
