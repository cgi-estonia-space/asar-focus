#pragma once


#include "device_padded_image.cuh"
#include "sar_metadata.h"


void AzimuthMultiLook(const DevicePaddedImage& d_slc, CudaWorkspace& out, SARMetadata& sar_meta);