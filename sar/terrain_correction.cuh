#pragma once


#include "cuda_util/cuda_workplace.h"
#include "cuda_util/device_padded_image.cuh"

#include "sar/sar_metadata.h"


struct TCResult
{
    int x_size;
    int y_size;
    float* d_data;
    double x_start;
    double y_start;
    double x_spacing;
    double y_spacing;
};

struct DEM
{
    int16_t* d_data;
    double x_orig;
    double y_orig;
};

TCResult RDTerrainCorrection(const DevicePaddedImage& d_img, DEM dem, const SARMetadata& sar_meta);