/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */
#include "parse_util.h"

#include "gmock/gmock.h"

namespace {

using ::testing::Eq;

using namespace alus::asar::envformat::parseutil;

TEST(ParseUtil, MaxValueForBits) {
    EXPECT_THAT((MaxValueForBits<uint8_t, 0>()), Eq(0));
    EXPECT_THAT((MaxValueForBits<uint8_t, 1>()), Eq(1));
    EXPECT_THAT((MaxValueForBits<uint16_t, 14>()), Eq(16383));
    EXPECT_THAT((MaxValueForBits<uint32_t, 14>()), Eq(16383));
    EXPECT_THAT((MaxValueForBits<uint64_t, 14>()), Eq(16383));
    // EXPECT_THAT((alus::asar::envformat::parseutil::MaxValueForBits<uint16_t, 20>()), Eq(16383)); - fails to compile
    EXPECT_THAT((MaxValueForBits<uint64_t, 40>()), Eq(1099511627775));
}

TEST(ParseUtil, CounterGapReturnsCorrectResults) {
    // In case of Envisat IM packets' sequence control field.
    constexpr auto PACKET_CTRL_MAX = MaxValueForBits<uint16_t, 14>();

    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(PACKET_CTRL_MAX, PACKET_CTRL_MAX)), Eq(0));
    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(15, 15)), Eq(0));
    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(0, 0)), Eq(0));

    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(0, 1)), Eq(1));
    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(0, PACKET_CTRL_MAX)), Eq(PACKET_CTRL_MAX));

    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(PACKET_CTRL_MAX, 0)), Eq(1));
    EXPECT_THAT((CounterGap<uint16_t, PACKET_CTRL_MAX>(PACKET_CTRL_MAX, 2)), Eq(3));
}

}  // namespace