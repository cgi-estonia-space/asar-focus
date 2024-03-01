

#include "patc.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include "bswap_util.h"
#include "date_time_util.h"
#include "filesystem_util.h"

namespace {

alus::asar::envformat::aux::Patc ConstructPatc(const std::vector<char>& buffer) {
    alus::asar::envformat::aux::Patc patc;

    patc.cyclic_counter = std::stoul(std::string(buffer.data() + 15, 4));
    patc.mission[0] = buffer.at(20);
    patc.mission[1] = buffer.at(21);
    patc.orbit_number = std::stoul(std::string(buffer.data() + 30, 5));
    memcpy(&patc.epoch_1950, buffer.data() + 38, 4);
    memcpy(&patc.microseconds, buffer.data() + 42, 4);

    auto epoch = alus::util::date_time::YYYYMMDD("19500101");
    std::cout << to_simple_string(epoch) << std::endl;
    epoch += boost::posix_time::seconds(patc.epoch_1950);
    std::cout << to_simple_string(epoch) << std::endl;
    epoch += boost::posix_time::microseconds(patc.microseconds);
    std::cout << to_simple_string(epoch) << std::endl;

    memcpy(&patc.sbt_counter, buffer.data() + 46, 4);
    memcpy(&patc.sbt_period, buffer.data() + 50, 4);

    return patc;
}

}  // namespace

namespace alus::asar::envformat::aux {

Patc ParsePatc(const std::filesystem::path& file_loc) {
    const auto& patc_buffer = util::filesystem::ReadAsBuffer(file_loc);

    if (patc_buffer.size() != 54) {
        throw std::runtime_error("PATC file '" + file_loc.string() + "' should be 54 bytes, actual - " +
                                 std::to_string(patc_buffer.size()));
    }

    return ConstructPatc(patc_buffer);
}

}  // namespace alus::asar::envformat::aux