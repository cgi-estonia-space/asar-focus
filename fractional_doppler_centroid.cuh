#pragma once

#include <complex>

#include "asar_metadata.h"

#include "device_padded_image.cuh"


void CalculateDopplerCentroid(const DevicePaddedImage& d_img, double prf, double& doppler_centroid);