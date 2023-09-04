

#pragma once

#include <complex>
#include <vector>

#include "cuda_util/cuda_workplace.h"
#include "cuda_util/device_padded_image.cuh"
#include "sar_chirp.h"
//#include "palsar_metadata.h"

// inplace range FFT -> mathed filtering in frequency domain with reference chirp -> inplace IFFT
void RangeCompression(DevicePaddedImage& data, const std::vector<std::complex<float>>& chirp_data, int chirp_samples,
                      CudaWorkspace& d_workspace);

// Secondary range compression - method 2
void SecondaryRangeCompression(DevicePaddedImage& data, const SARMetadata& sar_meta,
                               CudaWorkspace& d_workspace);
