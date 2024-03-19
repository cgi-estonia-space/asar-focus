#pragma once

struct RangeSpreadingLossParams {
    double slant_range_first;
    double range_spacing;
    double reference_range;
};

void RangeSpreadingLossCorrection(float* d_img_data, int range_size, int azimuth_size, RangeSpreadingLossParams params);