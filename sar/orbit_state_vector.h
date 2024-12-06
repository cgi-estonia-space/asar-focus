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

#include <string_view>
#include <vector>

#include <boost/date_time/posix_time/ptime.hpp>

struct OrbitStateVector {
    boost::posix_time::ptime time;
    double x_pos;
    double y_pos;
    double z_pos;
    double x_vel;
    double y_vel;
    double z_vel;
};

OrbitStateVector InterpolateOrbit(const std::vector<OrbitStateVector> &osv, boost::posix_time::ptime time);

inline double CalcVelocity(const OrbitStateVector &osv) {
    return sqrt(osv.x_vel * osv.x_vel + osv.y_vel * osv.y_vel + osv.z_vel * osv.z_vel);
}
