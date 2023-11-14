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

// dest_buffer must be on device and have enough capacity to accommodate original image (complex float) without padding.
void ConditionResults(DevicePaddedImage& img, IQ16* dest_buffer, float calibration_constant);