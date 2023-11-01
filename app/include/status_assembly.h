/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#pragma once

namespace alus::asar::status {

enum EXIT_CODE : int {
    SUCCESS = 0,
    INVALID_ARGUMENT = 1,
    UNSUPPORTED_DATASET = 2,
    INVALID_DATASET = 3,
    ASSERT_FAILURE = 4
};

}