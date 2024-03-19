#pragma once

#include "sar_metadata.h"

void SlantRangeToGroundRangeInterpolation(SARMetadata& sar_meta, const float* d_slant_in, float* d_gr_out);