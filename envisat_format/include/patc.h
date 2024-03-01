
#pragma once

#include <cstdint>
#include <filesystem>

namespace alus::asar::envformat::aux {

struct Patc {
    uint16_t cyclic_counter;
    char mission[2];
    uint32_t orbit_number;
    int32_t epoch_1950;
    int32_t microseconds;
    uint32_t sbt_counter;
    uint32_t sbt_period;  // Tick time value in picoseconds
};

Patc ParsePatc(const std::filesystem::path& file_loc);

}  // namespace alus::asar::envformat::aux