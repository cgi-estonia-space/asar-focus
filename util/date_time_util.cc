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
    boost::posix_time::ptime YYYYMMDD(std::string str) {
        auto* facet = new boost::posix_time::time_input_facet("%Y%m%d");
        std::stringstream date_stream(str);
        date_stream.imbue(std::locale(std::locale::classic(), facet));
        boost::posix_time::ptime time;
        date_stream >> time;
        return time;
    }
}