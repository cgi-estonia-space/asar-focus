#pragma once

#include "cuda_util/device_padded_image.cuh"

#include "sar_metadata.h"

struct CorrectionParams {
    size_t n_total_samples;
    // int n_sm;
};
void RawDataCorrection(DevicePaddedImage& img, CorrectionParams par, SARResults& results);