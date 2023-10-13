/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#pragma once

#include <string>

#include <boost/date_time/posix_time/ptime.hpp>

namespace alus::util::date_time {

    boost::posix_time::ptime YYYYMMDD(std::string str);
    boost::posix_time::ptime ParseDate(const std::string &date_string, std::locale format);
}