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

#include "cuda_util/device_padded_image.cuh"
#include "envisat_format/include/envisat_types.h"

namespace alus::asar::envformat {

// dest_buffer must be on device and have enough capacity to accommodate original image (complex float) without padding.
void ConditionResultsSLC(DevicePaddedImage& img, char* dest_space, size_t record_header_size,
                         float calibration_constant);
void ConditionResultsDetected(const float* d_img, char* d_dest_space, int range_size, int azimuth_size,
                              size_t record_header_size, float calibration_constant);

void ConvertErsImSamplesToComplex(const uint8_t* samples, size_t sample_count_per_range, size_t packets,
                                  const uint16_t* swst_codes, uint16_t min_swst, cufftComplex* gpu_buffer,
                                  size_t target_range_padded_samples);

void ConvertAsarImBlocksToComplex(const uint8_t* block_samples, size_t block_samples_item_length_bytes, size_t records,
                                  const uint16_t* swst_codes, uint16_t min_swst, cufftComplex* gpu_buffer,
                                  size_t target_range_padded_samples, const float* i_lut_fbaq4,
                                  const float* q_lut_fbaq4, size_t lut_items);

}  // namespace alus::asar::envformat
