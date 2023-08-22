#pragma once

#include "cuda_workplace.h"
#include "device_padded_image.cuh"

#include "asar_lvl0_parser.h"


/**
 * Basic RDA implementation as described in "Digital Processing of Synthetic Aperture Radar Data"
 * RCMC use no interpolator/nearest neighbor
 */
void RangeDopplerAlgorithm(const SARMetadata& metadata, DevicePaddedImage& src_img,
                               DevicePaddedImage& out_img, CudaWorkspace d_workspace);
