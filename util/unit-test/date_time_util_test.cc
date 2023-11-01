/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
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