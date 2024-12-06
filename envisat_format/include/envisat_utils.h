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

#pragma once

#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "checks.h"
#include "envisat_types.h"


template <size_t N>
void CopyStrPad(uc (&arr)[N], const std::string& s) {
    CHECK_BOOL(N >= s.size());
    memcpy(&arr[0], s.data(), s.size());

    size_t n_pad = N - s.size();
    if (n_pad) {
        memset(&arr[s.size()], ' ', n_pad);
    }
}

inline boost::posix_time::ptime MjdToPtime(mjd m) {
    boost::gregorian::date d(2000, 1, 1);

    d += boost::gregorian::days(m.days);
    boost::posix_time::ptime t(d);

    t += boost::posix_time::seconds(m.seconds);
    t += boost::posix_time::microseconds(m.micros);
    return t;
}

inline mjd PtimeToMjd(boost::posix_time::ptime in) {
    boost::gregorian::date d_0(2000, 1, 1);
    boost::gregorian::date d_1(in.date());

    mjd r = {};
    r.days = (d_1 - d_0).days();
    r.seconds = in.time_of_day().total_seconds();
    r.micros = in.time_of_day().total_microseconds() % 1000000;
    return r;
}

inline boost::posix_time::ptime StrToPtime(std::string str) {
    boost::posix_time::time_input_facet* facet =
        new boost::posix_time::time_input_facet("%d-%b-%Y %H:%M:%S%f");  //.%f");
    std::stringstream date_stream(str);
    date_stream.imbue(std::locale(std::locale::classic(), facet));
    boost::posix_time::ptime time;
    date_stream >> time;
    if (time.is_not_a_date_time()) {
        exit(1);
    }
    return time;
}

inline mjd MjdAddMicros(const mjd& d, int64_t micros_to_add) {
    auto ticktime = MjdToPtime(d);
    ticktime += boost::posix_time::microseconds(micros_to_add);
    return PtimeToMjd(ticktime);
}
