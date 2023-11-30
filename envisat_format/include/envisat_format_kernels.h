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
#include "envisat_format/include/envisat_types.h"

namespace alus::asar::envformat {

// dest_buffer must be on device and have enough capacity to accommodate original image (complex float) without padding.
void ConditionResults(DevicePaddedImage& img, char* dest_space, size_t record_header_size, float calibration_constant);

void ConvertErsImSamplesToComplex(uint8_t* samples, size_t source_range_samples, size_t packets, uint16_t* swst_codes,
                                  uint16_t min_swst, cufftComplex* gpu_buffer, size_t target_range_padded_samples);

void ConvertAsarImBlocksToComplex(uint8_t* block_samples, size_t block_samples_item_length_bytes, size_t records,
                                  uint16_t* swst_codes, uint16_t min_swst, cufftComplex* gpu_buffer,
                                  size_t target_range_padded_samples, float* i_lut_fbaq4, float* q_lut_fbaq4,
                                  size_t lut_items);

}  // namespace alus::asar::envformat
