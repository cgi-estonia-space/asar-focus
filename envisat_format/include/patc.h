
#pragma once

#include <cstdint>
#include <filesystem>

namespace alus::asar::envformat::aux {

// Should not be under this module. An exception, as a surprise, there is a file format
// which is NOT packaged into envisat format, nice.
struct Patc {
    uint16_t cyclic_counter;
    char mission[2];
    uint32_t orbit_number;
    int32_t epoch_1950; // Days
    int32_t milliseconds;
    uint32_t sbt_counter;
    uint32_t sbt_period;  // Tick time value in nanoseconds
};

Patc ParsePatc(const std::filesystem::path& file_loc);

}  // namespace alus::asar::envformat::aux