/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "envisat_parse_util.h"

#include "gmock/gmock.h"

namespace {

int Fbaq4IndexLut(int block, int idx) {
    switch (idx) {
        case 0b1111:
            idx = 0;
            break;
        case 0b1110:
            idx = 1;
            break;
        case 0b1101:
            idx = 2;
            break;
        case 0b1100:
            idx = 3;
            break;
        case 0b1011:
            idx = 4;
            break;
        case 0b1010:
            idx = 5;
            break;
        case 0b1001:
            idx = 6;
            break;
        case 0b1000:
            idx = 7;
            break;
        case 0b0000:
            idx = 8;
            break;
        case 0b0001:
            idx = 9;
            break;
        case 0b0010:
            idx = 10;
            break;
        case 0b0011:
            idx = 11;
            break;
        case 0b0100:
            idx = 12;
            break;
        case 0b0101:
            idx = 13;
            break;
        case 0b0110:
            idx = 14;
            break;
        case 0b0111:
            idx = 15;
            break;
    }

    return 256 * idx + block;
}

using ::testing::Eq;

TEST(EnvisatParseUtil, Fbaq4Index) {
    for (int i = 0; i < 16; i++) {
        EXPECT_THAT(alus::asar::envformat::Fbaq4Index(1, i), Eq(Fbaq4IndexLut(1, i)));
    }
}

}  // namespace
