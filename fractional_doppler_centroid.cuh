#pragma once

#include <complex>

//#include "asar_metadata.h"

#include "device_padded_image.cuh"


std::vector<double> CalculateDopplerCentroid(const DevicePaddedImage& d_img, double prf);