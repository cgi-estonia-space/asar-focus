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

#include "az_multilook.cuh"

#include "checks.h"

// TODO implement frequency domain multilooking? Is is better

// Implements sliding window azimuth time domain multilooking and azimuth resampling in one step
__global__ void TimeDomainAzimuthLook(const cufftComplex* data_in, int x_size, int in_y_size, int out_y_size,
                                      float resample_ratio, float* data_out) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int out_idx = y * x_size + x;

    if (x >= x_size || y >= out_y_size) {
        return;
    }

    float in_idx = y * resample_ratio;
    float mu = in_idx - static_cast<int>(in_idx);
    int cnt0 = 0;
    float sum0 = 0;
    {
        int y_start = in_idx - 2;
        int y_end = in_idx + 2;
        y_start = std::max(y_start, 0);
        y_end = std::min(y_end, in_y_size);
        for (int i = y_start; i < y_end; i++) {
            const int idx = i * x_size + x;
            const cufftComplex val = data_in[idx];
            sum0 += val.x * val.x + val.y * val.y;
            cnt0++;
        }
        if (cnt0) {
            sum0 /= cnt0;
        }
    }

    int cnt1 = 0;
    float sum1 = 0;
    {
        int y_start = in_idx - 1;
        int y_end = in_idx + 3;
        y_start = std::max(y_start, 0);
        y_end = std::min(y_end, in_y_size);
        for (int i = y_start; i < y_end; i++) {
            const int idx = i * x_size + x;
            const cufftComplex val = data_in[idx];
            sum1 += val.x * val.x + val.y * val.y;
            cnt1++;
        }
        if (cnt1) {
            sum1 /= cnt1;
        }
    }

    float interp_val = 0.0f;
    if (cnt0 && cnt1) {
        interp_val = sum0 + (sum1 - sum0) * mu;
    }

    data_out[out_idx] = interp_val;
}

void AzimuthMultiLook(const DevicePaddedImage& d_slc, CudaWorkspace& out, SARMetadata& sar_meta) {
    constexpr float final_spacing = 12.5f;  // IMP mode azimuth spacing

    const float ratio = final_spacing / sar_meta.azimuth_spacing;
    const int in_az_size = d_slc.YSize();
    const int out_az_size = std::round(d_slc.YSize() / ratio);
    const int range_size = d_slc.XSize();
    CHECK_BOOL(d_slc.XSize() == d_slc.XStride());
    dim3 block_size(16, 16);
    dim3 grid_size((range_size + 15) / 16, (out_az_size + 15) / 16);
    TimeDomainAzimuthLook<<<grid_size, block_size>>>(d_slc.Data(), range_size, in_az_size, out_az_size, ratio,
                                                     out.GetAs<float>());

    sar_meta.azimuth_spacing *= ratio;
    sar_meta.line_time_interval *= ratio;
    sar_meta.img.azimuth_size = out_az_size;

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}