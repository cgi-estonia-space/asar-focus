

#pragma once

#include <complex>
#include <vector>

#include "sar_chirp.h"
#include "cuda_workplace.h"
#include "device_padded_image.cuh"
//#include "palsar_metadata.h"

// inplace range FFT -> mathed filtering in frequency domain with reference chirp -> inplace IFFT
void RangeCompression(DevicePaddedImage& data, const std::vector<std::complex<float>>& chirp_data, int chirp_samples,
                          CudaWorkspace& d_workspace);

