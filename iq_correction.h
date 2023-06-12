#pragma once

#include <vector>
#include <complex>

struct IQ8
{
    uint8_t i;
    uint8_t q;
};

void iq_correct(const std::vector<IQ8> in, std::vector<std::complex<float>>& out, int rg_size, int rg_stride, int az_size, int az_stride);


