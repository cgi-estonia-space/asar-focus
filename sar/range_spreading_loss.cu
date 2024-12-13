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

#include "range_spreading_loss.cuh"

#include <fmt/format.h>
#include <cmath>
#include <vector>

#include "alus_log.h"
#include "checks.h"
#include "cuda_util.h"
#include "cuda_workplace.h"

__global__ void CorrectionKernel(float* d_img, int x_size, int y_size, const float* d_correction) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= x_size || y >= y_size) {
        return;
    }

    const int idx = y * x_size + x;

    d_img[idx] *= d_correction[x];
}

void RangeSpreadingLossCorrection(float* d_img_data, int range_size, int azimuth_size,
                                  RangeSpreadingLossParams params) {
    // 9.6 Range Spreading Loss Correction
    // S1-TN-MDA-52-7445_Sentinel-1 Level 1 Detailed Algorithm Definition_v2-5.pdf
    std::vector<float> scaling;
    for (int i = 0; i < range_size; i++) {
        const double slant_range = params.slant_range_first + i * params.range_spacing;
        const double ratio = params.reference_range / slant_range;
        const double val = pow(ratio, 3);
        scaling.push_back(1.0 / sqrt(val));
    }

    LOGI << fmt::format("Range spreading loss correction near-far = [{} {}]", scaling.front(), scaling.back());

    dim3 block_size(16, 16);
    dim3 grid_size((range_size + 15) / 16, (azimuth_size + 15) / 16);

    const size_t byte_size = range_size * sizeof(scaling[0]);
    const float* h_correction = scaling.data();
    CudaWorkspace d_buffer(byte_size);
    float* d_correction = d_buffer.GetAs<float>();
    CHECK_CUDA_ERR(cudaMemcpy(d_correction, h_correction, byte_size, cudaMemcpyHostToDevice));

    CorrectionKernel<<<grid_size, block_size>>>(d_img_data, range_size, azimuth_size, d_correction);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}