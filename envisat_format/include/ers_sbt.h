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

#include <cstdint>

#include "envisat_types.h"
#include "patc.h"

namespace alus::asar::envformat::ers {

void RegisterPatc(const aux::Patc& patc);
void AdjustIspSensingTime(mjd& isp_sensing_time, uint32_t sbt, uint32_t sbt_repeat, bool overflow = false);

}  // namespace alus::asar::envformat::ers