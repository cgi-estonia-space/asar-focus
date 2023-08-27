
#pragma once

#include <array>
#include <map>

#include <cufft.h>


// Ease of use wrappers around cuFFT plan APIs

// 2D image C2C float 1D FFT on each row(aka range direction)
    [[nodiscard]] cufftHandle PlanRangeFFT(int range_size, int azimuth_size, bool auto_alloc);

// 2D image C2C float 1D FFT on each column(aka azimuth direction)
    [[nodiscard]] cufftHandle PlanAzimuthFFT(int range_size, int azimuth_size, int range_stride, bool auto_alloc);

// 2D image C2C float 2D FFT
    [[nodiscard]] cufftHandle Plan2DFFT(int range_size, int azimuth_size);

    size_t EstimateRangeFFT(int range_size, int azimuth_size);

    size_t EstimateAzimuthFFT(int range_size, int azimuth_size, int range_stride);

    size_t Estimate2DFFT(int range_size, int azimuth_size);

    std::map<int, std::array<int, 4>> GetOptimalFFTSizes();

