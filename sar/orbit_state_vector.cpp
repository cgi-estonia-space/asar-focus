#include "orbit_state_vector.h"

#include <string_view>

#include <boost/date_time/posix_time/posix_time.hpp>

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
