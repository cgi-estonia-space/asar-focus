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
#include "sar_chirp.h"

#include "cufft_plan.h"
#include "cuda_util.h"
#include "cuda_cleanup.h"
#include "cufft_checks.h"


namespace
{
void ApplyHammingWindow(std::vector<std::complex<float>>& data) {
    const size_t N = data.size();
    for (size_t i = 0; i < N; i++) {
        size_t n = i;
        double term = (2 * M_PI * n) / N;
        double m = 0.54 - 0.46 * cos(term);
        data[i] *= m;
    }
}

void ScaleWindowed(std::vector<std::complex<float>>& chirp_data)
{
    cufftHandle plan = {};
    CufftPlanCleanup fft_cleanup(plan);
    size_t fft_size = chirp_data.size();
    // waste to use cuda for 1 small FFT, however no CPU FFT library has been integrated to the project
    CHECK_CUFFT_ERR(cufftPlan1d(&plan, fft_size, CUFFT_C2C, 1));
    cufftComplex* d_chirp = nullptr;
    size_t byte_sz = fft_size * sizeof(cufftComplex);
    CHECK_CUDA_ERR(cudaMalloc(&d_chirp, byte_sz));
    CudaMallocCleanup mem_cleanup(d_chirp);
    CHECK_CUDA_ERR(cudaMemcpy(d_chirp, chirp_data.data(), byte_sz, cudaMemcpyHostToDevice));
    CHECK_CUFFT_ERR(cufftExecC2C(plan, d_chirp, d_chirp, CUFFT_FORWARD));

    std::vector<std::complex<float>> h_fft(fft_size);
    CHECK_CUDA_ERR(cudaMemcpy(h_fft.data(), d_chirp, byte_sz, cudaMemcpyDeviceToHost));

    // total power in frequency bins, TODO(priit) equivalent in time domain?
    double scale = 0.0;
    for (const auto& el : h_fft) {
        float mag = std::abs(el);
        scale += mag * mag;
    }

    scale /= fft_size;

    // scale time td chirp
    for (auto& el : chirp_data) {
        el /= sqrt(scale);
    }
}
}

void GenerateChirpData(ChirpInfo& chirp, size_t padding_size) {
    double Kr = chirp.Kr;

    const int64_t n_samples = chirp.n_samples;
    const double dt = 1.0 / chirp.range_sampling_rate;

    chirp.time_domain.clear();
    chirp.time_domain.resize(n_samples);

    for (int64_t i = 0; i < n_samples; i++) {
        const double t = (i - n_samples / 2) * dt;
        const double phase = M_PI * Kr * t * t;
        std::complex<float> iq(cos(phase), sin(phase));

        chirp.time_domain[i] = iq;
    }

    chirp.padded_windowed_data = chirp.time_domain;

    if(chirp.apply_window)
    {
        ApplyHammingWindow(chirp.padded_windowed_data);
    }

    chirp.padded_windowed_data.resize(padding_size);

    // ApplyWindowPad(chirp_data, padding_size);
    if (chirp.apply_window) {
        ScaleWindowed(chirp.padded_windowed_data);
    } else {
        for (auto& el : chirp.padded_windowed_data) {
            el /= sqrt(n_samples);
        }
    }

}