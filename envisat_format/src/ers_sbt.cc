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

void AdjustIspSensingTime(mjd& isp_sensing_time, uint32_t sbt, uint32_t) {
    if (!patc_cor_set) {
        return;
    }

    boost::posix_time::ptime adjusted_ptime;
    int64_t delta_count{};
    if (sbt < patc_cor.sbt_counter) {
        delta_count = static_cast<int64_t>(sbt) - patc_cor.sbt_counter;
    } else {
        delta_count = parseutil::CounterGap<uint32_t, SBT_MAX>(patc_cor.sbt_counter, sbt);
    }
    const int64_t picoseconds_diff = (patc_cor.sbt_period * 1e3) * delta_count;
    const auto microseconds_val = static_cast<int64_t>(picoseconds_diff / 1e6);
    adjusted_ptime = patc_time_point + boost::posix_time::microseconds(microseconds_val);

    isp_sensing_time = PtimeToMjd(adjusted_ptime);
}

}  // namespace alus::asar::envformat::ers
