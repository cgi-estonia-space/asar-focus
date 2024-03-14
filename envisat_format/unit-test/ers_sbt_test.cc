

#include "ers_sbt.h"

#include <stdexcept>

#include "gmock/gmock.h"

#include "envisat_types.h"
#include "patc.h"

namespace {

using ::testing::Eq;

using namespace alus::asar::envformat::ers;
using namespace alus::asar::envformat::aux;

TEST(ErsSbt, WhenEpochIsNegativeRegisterPatcThrows) {
    {
        Patc p{};
        p.milliseconds = -1023;
        p.epoch_1950 = 365 * 50;
        EXPECT_THROW(RegisterPatc(p), std::runtime_error);
    }
    {
        Patc p{};
        p.milliseconds = 1000;
        p.epoch_1950 = -1;
        EXPECT_THROW(RegisterPatc(p), std::runtime_error);
    }
}

TEST(ErsSbt, WhenEpochIsEarlierThan1990RegisterPatcThrows) {
    Patc p{};
    p.epoch_1950 = 14974;
    EXPECT_THROW(RegisterPatc(p), std::runtime_error);
}

TEST(ErsSbt, WhenPatcIsNotRegisteredDoesNotAdjustIspSensingTime) {
    const mjd d_const{1, 1, 1};
    mjd d = d_const;
    AdjustIspSensingTime(d, 1000, 0);
    ASSERT_THAT(d, Eq(d_const));
}

TEST(ErsSbt, TimeIsCorrectByAdjustIspSensingTime) {
    // All the values in this test has been copied from the TDS01.
    Patc p{};
    p.cyclic_counter = 720;
    p.mission[0] = 'E';
    p.mission[1] = '1';
    p.orbit_number = 929;
    p.epoch_1950 = 15236;
    p.milliseconds = 77966950;
    p.sbt_counter = 1440813490;
    p.sbt_period = 3906249;

    RegisterPatc(p);

    // SBT Counter earlier than in PATC
    {
        mjd isp_adjusted = {-3026, 77935, 870958};
        const uint32_t sbt{1440805534};
        AdjustIspSensingTime(isp_adjusted, sbt, 0);
        const mjd isp_expected = {-3026, 77935, 871883};
        ASSERT_THAT(isp_adjusted, Eq(isp_expected))
            << "days:    " << isp_adjusted.days << "|" << isp_expected.days << std::endl
            << "seconds: " << isp_adjusted.seconds << "|" << isp_expected.seconds << std::endl
            << "micros:  " << isp_adjusted.micros << "|" << isp_expected.micros;
    }

    // SBT counter equal to the one in PATC
    {
        const mjd isp_expected = {-3026, 77966, 950000};
        const uint32_t sbt{1440813490};
        mjd isp_adjusted{};
        AdjustIspSensingTime(isp_adjusted, sbt, 0);
        ASSERT_THAT(isp_adjusted, Eq(isp_expected))
            << "days:    " << isp_adjusted.days << "|" << isp_expected.days << std::endl
            << "seconds: " << isp_adjusted.seconds << "|" << isp_expected.seconds << std::endl
            << "micros:  " << isp_adjusted.micros << "|" << isp_expected.micros;
    }

    // SBT Counter after the one in PATC
    {
        mjd isp_adjusted = {-3026, 77966, 960794};
        const uint32_t sbt{1440813493};
        AdjustIspSensingTime(isp_adjusted, sbt, 0);
        const mjd isp_expected = {-3026, 77966, 961718};
        ASSERT_THAT(isp_adjusted, Eq(isp_expected))
            << "days:    " << isp_adjusted.days << "|" << isp_expected.days << std::endl
            << "seconds: " << isp_adjusted.seconds << "|" << isp_expected.seconds << std::endl
            << "micros:  " << isp_adjusted.micros << "|" << isp_expected.micros;
    }

    // Same as "SBT count earlier" case, but with overflow condition.
    {
        mjd isp_adjusted = {-3026, 77935, 870958};
        const uint32_t sbt{1440805534};
        AdjustIspSensingTime(isp_adjusted, sbt, 0, true);
        const mjd isp_expected = {-2831, 7147, 576915};
        ASSERT_THAT(isp_adjusted, Eq(isp_expected))
            << "days:    " << isp_adjusted.days << "|" << isp_expected.days << std::endl
            << "seconds: " << isp_adjusted.seconds << "|" << isp_expected.seconds << std::endl
            << "micros:  " << isp_adjusted.micros << "|" << isp_expected.micros;
    }
}

}  // namespace