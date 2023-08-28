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

std::vector<OrbitStateVector>
FindOrbits(boost::posix_time::ptime start, boost::posix_time::ptime stop, std::string_view orbit_path);

OrbitStateVector InterpolateOrbit(const std::vector<OrbitStateVector> &osv, boost::posix_time::ptime time);

inline double CalcVelocity(const OrbitStateVector &osv) {
    return sqrt(osv.x_vel * osv.x_vel + osv.y_vel * osv.y_vel + osv.z_vel * osv.z_vel);
}
