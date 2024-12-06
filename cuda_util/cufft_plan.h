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

#include <array>
#include <map>

#include <cufft.h>

// Ease of use wrappers around cuFFT plan APIs

// 2D image C2C float 1D FFT on each row(aka range direction)
[[nodiscard]] cufftHandle PlanRangeFFT(int range_size, int azimuth_size, bool auto_alloc);

// 2D image C2C float 1D FFT on each column(aka azimuth direction)
[[nodiscard]] cufftHandle PlanAzimuthFFT(int range_size, int azimuth_size, int range_stride, bool auto_alloc);

// 2D image C2C float 2D FFT
[[nodiscard]] cufftHandle Plan2DFFT(int range_size, int azimuth_size);

size_t EstimateRangeFFT(int range_size, int azimuth_size);

size_t EstimateAzimuthFFT(int range_size, int azimuth_size, int range_stride);

size_t Estimate2DFFT(int range_size, int azimuth_size);

std::map<int, std::array<int, 4>> GetOptimalFFTSizes();
