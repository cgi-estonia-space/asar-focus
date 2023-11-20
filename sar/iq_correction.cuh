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

#include "cuda_util/device_padded_image.cuh"
#include "sar_metadata.h"

struct CorrectionParams {
    size_t n_total_samples;
    // int n_sm;
};
void RawDataCorrection(DevicePaddedImage& img, CorrectionParams par, SARResults& results);
