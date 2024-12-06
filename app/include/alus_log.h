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
