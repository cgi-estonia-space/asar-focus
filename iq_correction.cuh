#pragma once


#include "device_padded_image.cuh"


#include "asar_lvl0_parser.h"


struct CorrectionParams
{
    size_t n_total_samples;
    //int n_sm;
};
void RawDataCorrection(DevicePaddedImage& img, CorrectionParams par, SARResults& results);