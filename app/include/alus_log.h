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

#include <boost/log/trivial.hpp>

namespace alus::asar::log {

enum class Level { VERBOSE, DEBUG, INFO, WARNING, ERROR };
enum class Format { DEFAULT };

void Initialize(Format f = Format::DEFAULT);
void SetLevel(Level level);

}  // namespace alus::asar::log

#define LOGV BOOST_LOG_TRIVIAL(trace)
#define LOGD BOOST_LOG_TRIVIAL(debug)
#define LOGI BOOST_LOG_TRIVIAL(info)
#define LOGW BOOST_LOG_TRIVIAL(warning)
#define LOGE BOOST_LOG_TRIVIAL(error)
