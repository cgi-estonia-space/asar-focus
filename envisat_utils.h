#pragma once

#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "checks.h"

#include "envisat_utils.h"




template<size_t N>
void CopyStrPad(uc (&arr)[N], const std::string &s) {
    CHECK_BOOL(N >= s.size());
    memcpy(&arr[0], s.data(), s.size());

    size_t n_pad = N - s.size();
    if (n_pad) {
        memset(&arr[s.size()], ' ', n_pad);
    }
}



inline boost::posix_time::ptime MjdToPtime(mjd m)
{
    boost::gregorian::date d(2000, 1, 1);

    d += boost::gregorian::days(m.days);
    boost::posix_time::ptime t(d);

    t += boost::posix_time::seconds(m.seconds);
    t += boost::posix_time::microseconds (m.micros);
    return t;
}

inline mjd PtimeToMjd(boost::posix_time::ptime in)
{

    boost::gregorian::date d_0(2000, 1, 1);

    boost::gregorian::date d_1(in.date());

    std::cout << "SIIN : \n\n" << in << "\n\n";

    mjd r = {};
    r.days = (d_1 - d_0).days();
    r.seconds = in.time_of_day().total_seconds();
    r.micros = in.time_of_day().total_microseconds() % 1000000;
    return r;
}
