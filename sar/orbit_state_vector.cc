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
