#pragma once

#include "cuda_util/cuda_workplace.h"
#include "cuda_util/device_padded_image.cuh"

#include "sar/sar_metadata.h"

/**
 * Basic RDA implementation as described in "Digital Processing of Synthetic Aperture Radar Data"
 * RCMC use no interpolator/nearest neighbor
 */
void RangeDopplerAlgorithm(const SARMetadata& metadata, DevicePaddedImage& src_img, DevicePaddedImage& out_img,
                           CudaWorkspace& d_workspace);
