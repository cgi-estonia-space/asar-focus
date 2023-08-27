

/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */

#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include <cuComplex.h>

constexpr double SOL = 299792458;

inline int NextPower2(int value) {
    int r = 1;
    while (r < value) {
        r *= 2;
    }
    return r;
}

// https://en.wikipedia.org/wiki/Hann_function
inline void ApplyHanningWindow(std::vector<std::complex<float>>& data) {
    const size_t N = data.size();
    for (size_t i = 0; i < N; i++) {
        size_t n = i;
        double term = (2 * M_PI * n) / N;
        double m = 0.5 * (1 - cos(term));
        data[i] *= m;
    }
}

inline void ApplyHammingWindow(std::vector<std::complex<float>>& data) {
    const size_t N = data.size();
    for (size_t i = 0; i < N; i++) {
        size_t n = i;
        double term = (2 * M_PI * n) / N;
        double m = 0.54 - 0.46 * cos(term);
        data[i] *= m;
    }
}

inline void InplaceComplexToIntensity(cuComplex* data, size_t n) {
    for (size_t idx = 0; idx < n; idx++) {
        float i = data[idx].x;
        float q = data[idx].y;

        float intensity = i * i + q * q;

        auto& dest = data[idx / 2];
        if ((idx % 2) == 0) {
            dest.x = intensity;
        } else {
            dest.y = intensity;
        }
    }
}

std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree);

inline double polyval(const double* p, int n, double x) {
    double val = 0.0;
    // Horner's method
    for (int i = 0; i < n; i++) {
        val *= x;
        val += p[i];
    }
    return val;
}
