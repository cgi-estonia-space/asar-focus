#pragma once

#include <memory>

#include "cuda_util/device_padded_image.cuh"
#include "envisat_format/include/envisat_types.h"
#include "sar_metadata.h"

struct CorrectionParams {
    size_t n_total_samples;
    // int n_sm;
};
void RawDataCorrection(DevicePaddedImage& img, CorrectionParams par, SARResults& results);

std::unique_ptr<IQ16[]> ResultsCorrection(DevicePaddedImage& img, CudaWorkspace& dest_space, float calibration_constant);