
#include "ers_tim.h"

#include "date_time_util.h"
#include "filesystem_util.h"

namespace {

constexpr std::string_view TIM_TIMESTAMP_PATTERN{"%d-%b-%Y %H:%M:%S%f"};

alus::asar::envformat::aux::Tim ConstructTim(const std::vector<char>& buffer) {
    const auto* timestamp_styling = new boost::posix_time::time_input_facet(TIM_TIMESTAMP_PATTERN.data());
    const auto locale = std::locale(std::locale::classic(), timestamp_styling);

    alus::asar::envformat::aux::Tim entry{};
    // Filled with spaces (0x20) before content is found in the bottom/end of the file.
    const auto utc_str = std::string(buffer.data() + 0x659, 27);
    entry.utc_point = alus::util::date_time::ParseDate(utc_str, locale);
    (void)utc_str;
    std::cout << to_simple_string(entry.utc_point) << std::endl;
    entry.sbt_counter = std::stoul(std::string(buffer.data() + 0x675, 11));
    entry.sbt_period = std::stoul(std::string(buffer.data() + 0x681, 11));

    return entry;
}

}  // namespace

namespace alus::asar::envformat::aux {

Tim ParseTim(const std::filesystem::path& file_loc) {
    const auto& tim_buffer = util::filesystem::ReadAsBuffer(file_loc);

    if (tim_buffer.size() != 1677) {
        throw std::runtime_error("TIM file '" + file_loc.string() + "' should be 1677 bytes, actual - " +
                                 std::to_string(tim_buffer.size()));
    }

    auto tim = ConstructTim(tim_buffer);
    const auto filename = file_loc.filename().string();
    tim.mission[0] = filename.front();
    tim.mission[1] = filename.at(2);
    tim.orbit_number = std::stoul(filename.data() + (filename.length() - 7));

    return tim;
}

}  // namespace alus::asar::envformat::aux