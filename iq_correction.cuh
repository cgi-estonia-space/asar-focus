#pragma once


#include "device_padded_image.cuh"


#include "asar_lvl0_parser.h"


struct IQ8
{
    uint8_t i;
    uint8_t q;
};

void iq_correct_cuda(const std::vector<IQ8>& vec, DevicePaddedImage& out);