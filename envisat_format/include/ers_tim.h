

#pragma once

#include <cstdint>
#include <filesystem>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace alus::asar::envformat::aux {

struct Tim {
    uint32_t orbit_number;
    char mission[2];  // Retrofit for compatibility with PATC
    boost::posix_time::ptime utc_point;
    uint32_t sbt_counter;
    uint32_t sbt_period;  // Tick time value in picoseconds
};

Tim ParseTim(const std::filesystem::path& file_loc);

}  // namespace alus::asar::envformat::aux