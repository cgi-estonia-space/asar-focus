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

/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */

#include "alus_log.h"

#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>  // Might be falsely flagged by CLion as unnecessary.
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>

namespace {
// Violating non-trivially destructible property. This is an exception.
const std::map<alus::asar::log::Level, boost::log::trivial::severity_level> boost_level_map{
    {alus::asar::log::Level::VERBOSE, boost::log::trivial::severity_level::trace},
    {alus::asar::log::Level::DEBUG, boost::log::trivial::severity_level::debug},
    {alus::asar::log::Level::INFO, boost::log::trivial::severity_level::info},
    {alus::asar::log::Level::WARNING, boost::log::trivial::severity_level::warning},
    {alus::asar::log::Level::ERROR, boost::log::trivial::severity_level::error}};

}  // namespace

namespace alus::asar::log {

void Initialize(Format) {
    SetLevel(Level::VERBOSE);
}

void SetLevel(Level level) {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost_level_map.at(level));
}
}  // namespace alus::asar::log
