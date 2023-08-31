#include "orbit_state_vector.h"

#include <string_view>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "envisat_format/doris_orbit.h"

std::vector<OrbitStateVector> FindOrbits(boost::posix_time::ptime start, boost::posix_time::ptime stop,
                                         std::string_view orbit_path) {
    // Assert that stop is greater than start.
    std::vector<OrbitStateVector> osv = {

    };

    auto parsable = alus::dorisorbit::Parsable::TryCreateFrom(orbit_path);
    auto orbit_state_vector = parsable.CreateOrbitInfo();

    const auto delta_minutes = boost::posix_time::minutes(5);
    const auto start_delta = start - delta_minutes;
    const auto end_delta = stop + delta_minutes;
    for (const auto& ov : orbit_state_vector) {
        if (ov.time > start_delta && ov.time < end_delta) {
            osv.push_back(ov);
        }
    }

    if (osv.size() < 8) {
        std::string msg = "Wrong date on orbit file?\nOSV start/end = ";
        msg += boost::posix_time::to_simple_string(orbit_state_vector.front().time) + " / ";
        msg += boost::posix_time::to_simple_string(orbit_state_vector.back().time);
        msg += " Sensing start = " + boost::posix_time::to_simple_string(start) + "\n";
        ERROR_EXIT(msg);
    }

    std::cout << start << "\n";
    std::cout << stop << "\n";
    std::cout << osv.front().time << "\n";
    std::cout << osv.back().time << "\n";
    CHECK_BOOL(start >= osv.front().time && stop <= osv.back().time);
    return osv;
}

OrbitStateVector InterpolateOrbit(const std::vector<OrbitStateVector>& osv, boost::posix_time::ptime time) {
    double time_point = (time - osv.front().time).total_microseconds() * 1e-6;

    OrbitStateVector r = {};
    r.time = time;
    const int n = osv.size();
    for (int i = 0; i < n; i++) {
        double mult = 1;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double xj = (osv.at(j).time - osv.front().time).total_microseconds() * 1e-6;
            double xi = (osv.at(i).time - osv.front().time).total_microseconds() * 1e-6;
            mult *= (time_point - xj) / (xi - xj);
        }

        r.x_pos += mult * osv[i].x_pos;
        r.y_pos += mult * osv[i].y_pos;
        r.z_pos += mult * osv[i].z_pos;
        r.x_vel += mult * osv[i].x_vel;
        r.y_vel += mult * osv[i].y_vel;
        r.z_vel += mult * osv[i].z_vel;
    }

    return r;
}