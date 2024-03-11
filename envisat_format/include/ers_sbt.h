
#pragma once

#include <cstdint>

#include "envisat_types.h"
#include "patc.h"

namespace alus::asar::envformat::ers {

void RegisterPatc(const aux::Patc& patc);
void AdjustIspSensingTime(mjd& isp_sensing_time, uint32_t sbt, uint32_t sbt_repeat);

}  // namespace alus::asar::envformat::ers