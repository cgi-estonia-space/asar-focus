/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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