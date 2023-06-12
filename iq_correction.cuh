#pragma once


#include "device_padded_image.cuh"

#include "iq_correction.h"

#include "asar_metadata.h"


void iq_correct_cuda(const std::vector<IQ8>& vec, DevicePaddedImage& out);