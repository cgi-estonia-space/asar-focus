#include "ers_sbt.h"

#include <cstdint>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "date_time_util.h"
#include "envisat_types.h"
#include "envisat_utils.h"
#include "parse_util.h"
#include "patc.h"

namespace {

bool patc_cor_set{false};
alus::asar::envformat::aux::Patc patc_cor{};
boost::posix_time::ptime patc_time_point;

constexpr auto SBT_MAX{alus::asar::envformat::parseutil::MaxValueForBits<uint32_t, 32>()};

}  // namespace

namespace alus::asar::envformat::ers {

void RegisterPatc(const aux::Patc& patc) {
    patc_cor = patc;
    const auto epoch = alus::util::date_time::YYYYMMDD("19500101");
    patc_time_point = epoch + boost::posix_time::time_duration(patc.epoch_1950 * 24, 0, 0, 0);
    patc_time_point += boost::posix_time::milliseconds(patc.milliseconds);
    patc_cor_set = true;
}

void AdjustIspSensingTime(mjd& isp_sensing_time, uint32_t sbt, uint32_t sbt_repeat) {
    if (!patc_cor_set) {
        return;
    }

    //    const auto sensing_time = MjdToPtime(isp_sensing_time);
    //    std::cout << to_simple_string(sensing_time) << std::endl;
    //    std::cout << to_simple_string(patc_time_point) << std::endl;
    //    if (sensing_time < patc_time_point) {
    //        return;
    //        throw std::runtime_error("Given ISP sensing time '" + to_simple_string(sensing_time) +
    //                                 "' is ahead of the supplied PATC synchronization point - '" +
    //                                 to_simple_string(patc_time_point) +
    //                                 "'. Expecting synchronization point to be earlier than packet sensing times.");
    //    }

    (void)sbt_repeat;
    // Not valid, once sbt reaches max and resets this would yield incorrect values.
    boost::posix_time::ptime adjusted_ptime;
    if (sbt < patc_cor.sbt_counter) {
        const auto delta_count = patc_cor.sbt_counter - sbt;
        //        throw std::runtime_error("Given PATC SBT value '" + std::to_string(patc_cor.sbt_counter) +
        //                                 "' is greater than the dataset's packet SBT value '" + std::to_string(sbt) +
        //                                 "'. Expecting synchronization point to be earlier than packets' SBT
        //                                 counter.");
        const int64_t picoseconds_elapsed = patc_cor.sbt_period * delta_count;
        adjusted_ptime =
            patc_time_point - boost::posix_time::microseconds(static_cast<uint32_t>(picoseconds_elapsed / 1e6));
    } else {
        const auto delta_count = parseutil::CounterGap<uint32_t, SBT_MAX>(patc_cor.sbt_counter, sbt);
        const int64_t picoseconds_elapsed = patc_cor.sbt_period * delta_count;
        adjusted_ptime =
            patc_time_point + boost::posix_time::microseconds(static_cast<uint32_t>(picoseconds_elapsed / 1e6));
    }
    isp_sensing_time = PtimeToMjd(adjusted_ptime);
}

}  // namespace alus::asar::envformat::ers
