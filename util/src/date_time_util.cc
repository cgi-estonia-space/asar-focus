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

#include <sstream>

#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

namespace alus::util::date_time {

boost::posix_time::ptime ParseDate(const std::string& date_string, std::locale format) {
    std::stringstream stream(date_string);
    stream.imbue(format);
    boost::posix_time::ptime date;
    stream >> date;

    return date;
}

boost::posix_time::ptime YYYYMMDD(std::string str) {
    auto* facet = new boost::posix_time::time_input_facet("%Y%m%d");
    std::stringstream date_stream(str);
    date_stream.imbue(std::locale(std::locale::classic(), facet));
    boost::posix_time::ptime time;
    date_stream >> time;
    return time;
}

boost::posix_time::ptime ToYyyyMmDd_HhMmSsFfffff(const std::string& date_string) {
    constexpr std::string_view format{"%Y%m%d_%H%M%S%F"};
    if (date_string.length() != 21 || date_string.at(8) != '_') {
        throw std::invalid_argument("Given date string '" + date_string + "' does not conform to a format - " +
                                    std::string(format));
    }
    // 20040229_212504912000
    // Not able to parse fractional seconds
    auto* facet =
        new boost::posix_time::time_input_facet(std::string(format.data(), format.data() + (format.length() - 2)));
    std::stringstream date_stream(date_string.substr(0, 21 - 6));
    date_stream.imbue(std::locale(std::locale::classic(), facet));
    boost::posix_time::ptime result_time;
    date_stream >> result_time;

    if (result_time.is_not_a_date_time()) {
        throw std::invalid_argument("Given date string '" + date_string + "' does not conform to a format - " +
                                    std::string(format));
    }

    result_time += boost::posix_time::microseconds(std::stoul(date_string.data() + 15));

    return result_time;
}
}  // namespace alus::util::date_time