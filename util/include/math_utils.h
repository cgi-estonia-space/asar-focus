/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cmath>
#include <complex>
#include <numeric>
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

std::vector<double> Polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree);

inline double Polyval(const double* p, int n, double x) {
    double val = 0.0;
    // Horner's method
    for (int i = 0; i < n; i++) {
        val *= x;
        val += p[i];
    }
    return val;
}

inline void PolyvalRange(const std::vector<double>& p, int x_start, int x_end, std::vector<double>& x,
                         std::vector<double>& y) {
    x.resize(x_end - x_start);
    std::iota(x.begin(), x.end(), x_start);
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = Polyval(p.data(), p.size(), x[i]);
    }
}
