

#pragma once

#include <optional>

#include "cuda_util/device_padded_image.cuh"

void WriteIntensityPaddedImg(const DevicePaddedImage& img, const char* path);
