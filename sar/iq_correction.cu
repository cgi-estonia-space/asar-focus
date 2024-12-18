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
#include "iq_correction.cuh"

#include <numeric>

#include "alus_log.h"
#include "checks.h"
#include "cuda_util/cuda_algorithm.cuh"
#include "cuda_util/cuda_cleanup.h"

namespace {

constexpr int REDUCE_BLOCK_SIZE = 1024;

/*
 * Simplified reduction kernels for I/Q summation,
 * n(should be small)of 1D thread blocks iterate over the image, summing I and Q data,
 * outputting the result this into a n sized device array.
 * Likely can improved via thrust::reduce or a more complete multistep reduction implementation.
 * Additionally intermediates are stored in double, potentially could be store in floats.
 * However the total runtime of these operations is likely neglible, thus no point optimizing here.
 */


__global__ void GainCorrectionReduction(const cufftComplex* src, int x_size, int y_size, float* i_result_arr,
                                        float* q_result_arr) {
    __shared__ float i_sq_shared_acc[REDUCE_BLOCK_SIZE];
    __shared__ float q_sq_shared_acc[REDUCE_BLOCK_SIZE];

    const int shared_idx = threadIdx.x;
    const int result_idx = blockIdx.y;
    const int y_start = blockIdx.y;
    const int x_start = threadIdx.x;
    const int y_step = gridDim.y;
    const int x_step = blockDim.x;

    double i_sq_sum = 0.0;
    double q_sq_sum = 0.0;
    for (int y = y_start; y < y_size; y += y_step) {
        for (int x = x_start; x < x_size; x += x_step) {
            int idx = (y * x_size) + x;
            auto sample = src[idx];
            float i = sample.x;
            float q = sample.y;

            if (!std::isnan(i) && !std::isnan(q)) {
                i_sq_sum += (i * i);
                q_sq_sum += (q * q);
            }
        }
    }

    // store each thread's result in shared memory, ensure all threads reach this point
    i_sq_shared_acc[shared_idx] = i_sq_sum;
    q_sq_shared_acc[shared_idx] = q_sq_sum;
    __syncthreads();

    // reduce the shared memory array to the first element
    // Reduction #3: Sequential Addressing from NVidia Optimizing Parallel Reduction
    for (int s = REDUCE_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            i_sq_shared_acc[shared_idx] += i_sq_shared_acc[shared_idx + s];
            q_sq_shared_acc[shared_idx] += q_sq_shared_acc[shared_idx + s];
        }
        __syncthreads();
    }

    // write the final result to global memory
    if (shared_idx == 0) {
        i_result_arr[result_idx] = i_sq_shared_acc[0];
        q_result_arr[result_idx] = q_sq_shared_acc[0];
    }
}

__global__ void PhaseCorrectionReduction(const cufftComplex* src, int x_size, int y_size, float* iq_result_arr,
                                         float* i_sq_result_arr) {
    __shared__ double iq_shared_acc[REDUCE_BLOCK_SIZE];
    __shared__ double i_sq_shared_acc[REDUCE_BLOCK_SIZE];

    const int shared_idx = threadIdx.x;
    const int result_idx = blockIdx.y;
    const int y_start = blockIdx.y;
    const int x_start = threadIdx.x;
    const int y_step = gridDim.y;
    const int x_step = blockDim.x;

    double iq_sum = 0.0;
    double i_sq_sum = 0.0;
    for (int y = y_start; y < y_size; y += y_step) {
        for (int x = x_start; x < x_size; x += x_step) {
            int idx = (y * x_size) + x;
            auto sample = src[idx];
            float i = sample.x;
            float q = sample.y;

            if (!std::isnan(i) && !std::isnan(q)) {
                iq_sum += i * q;
                i_sq_sum += i * i;
            }
        }
    }

    // store each thread's result in shared memory, ensure all threads reach this point
    iq_shared_acc[shared_idx] = iq_sum;
    i_sq_shared_acc[shared_idx] = i_sq_sum;
    __syncthreads();

    // reduce the shared memory array to the first element
    // Reduction #3: Sequential Addressing from NVidia Optimizing Parallel Reduction
    for (int s = REDUCE_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            iq_shared_acc[shared_idx] += iq_shared_acc[shared_idx + s];
            i_sq_shared_acc[shared_idx] += i_sq_shared_acc[shared_idx + s];
        }
        __syncthreads();
    }

    // write the final result to global memory
    if (shared_idx == 0) {
        iq_result_arr[result_idx] = iq_shared_acc[0];
        i_sq_result_arr[result_idx] = i_sq_shared_acc[0];
    }
}

__global__ void CorrectDCBias(cufftComplex* data, int x_size, int y_size, float dc_i, float dc_q) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int idx = x + y * x_size;

    if (x < x_size && y < y_size) {
        data[idx].x -= dc_i;
        data[idx].y -= dc_q;
    }
}

__global__ void ApplyGainCorrection(cufftComplex* data, int x_size, int y_size, float i_gain) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int idx = (y * x_size) + x;

    if (x < x_size && y < y_size) {
        data[idx].x /= i_gain;
    }
}

__global__ void ApplyPhaseCorrection(cufftComplex* data, int x_size, int y_size, float sin_corr, float cos_corr) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = (y * x_size) + x;

    auto pix = data[idx];
    float i = pix.x;
    float q = pix.y;

    float corr_q = (q - i * sin_corr) / cos_corr;

    if (x < x_size && y < y_size) {
        data[idx].y = corr_q;
    }
}

double SumSmallDeviceArray(float* d_data, int n) {
    CHECK_BOOL(n < 100);
    std::array<float, 100> h_data;
    CHECK_CUDA_ERR(cudaMemcpy(h_data.data(), d_data, n * sizeof(h_data[0]), cudaMemcpyDeviceToHost));
    return std::accumulate(h_data.begin(), h_data.begin() + n, 0.0);
}
}  // namespace

void RawDataCorrection(DevicePaddedImage& img, CorrectionParams par, SARResults& results) {
    int n_reduce_blocks = 50;  // should be based on sm count
    float* d_reduction_buffer1;
    float* d_reduction_buffer2;
    size_t red_byte_sz = sizeof(float) * n_reduce_blocks;
    CHECK_CUDA_ERR(cudaMalloc(&d_reduction_buffer1, red_byte_sz));
    CudaMallocCleanup clean1(d_reduction_buffer1);
    CHECK_CUDA_ERR(cudaMalloc(&d_reduction_buffer2, red_byte_sz));
    CudaMallocCleanup clean2(d_reduction_buffer2);
    const int y_size = img.YSize();
    const int x_size = img.XStride();

    {
        auto mean = img.CalcMean(par.n_total_samples);
        auto input_std_dev = img.CalcStdDev(mean.x, mean.y, par.n_total_samples);
        LOGD << "DC bias = " << mean.x << " " <<  mean.y;
        LOGD << "Input I/Q std dev = " << input_std_dev.x << " " << input_std_dev.y;

        results.dc_i = mean.x;
        results.dc_q = mean.y;
        results.stddev_i = input_std_dev.x;
        results.stddev_q = input_std_dev.y;

        {
            dim3 block_sz(16, 16);
            dim3 grid_sz((x_size + 15) / 16, (y_size + 15) / 16);
            CorrectDCBias<<<grid_sz, block_sz>>>(img.Data(), x_size, y_size, results.dc_i, results.dc_q);
        }
    }

    {
        // Gain correction

        double i_gain = 0.0;

        float* d_i_sq_sum = d_reduction_buffer1;
        float* d_q_sq_sum = d_reduction_buffer2;
        {
            dim3 block_sz(REDUCE_BLOCK_SIZE, 1);
            dim3 grid_sz(1, n_reduce_blocks);
            GainCorrectionReduction<<<grid_sz, block_sz>>>(img.Data(), x_size, y_size, d_i_sq_sum, d_q_sq_sum);
        }

        double i_sq_sum = SumSmallDeviceArray(d_i_sq_sum, n_reduce_blocks);
        double q_sq_sum = SumSmallDeviceArray(d_q_sq_sum, n_reduce_blocks);

        i_gain = sqrt(i_sq_sum / q_sq_sum);
        results.iq_gain = i_gain;

        {
            dim3 block_sz(16, 16);
            dim3 grid_sz((x_size + 15) / 16, (y_size + 15) / 16);
            ApplyGainCorrection<<<grid_sz, block_sz>>>(img.Data(), x_size, y_size, i_gain);
        }
        LOGD << "Gain imbalance = " << results.iq_gain;
    }

    {
        // Phase correction

        float* d_iq_sum = d_reduction_buffer1;
        float* d_i_sq_sum = d_reduction_buffer2;
        {
            dim3 block_sz(REDUCE_BLOCK_SIZE, 1);
            dim3 grid_sz(1, n_reduce_blocks);
            PhaseCorrectionReduction<<<grid_sz, block_sz>>>(img.Data(), x_size, y_size, d_iq_sum, d_i_sq_sum);
        }

        double iq_sum = SumSmallDeviceArray(d_iq_sum, n_reduce_blocks);
        double i_sq_sum = SumSmallDeviceArray(d_i_sq_sum, n_reduce_blocks);

        double delta_theta = iq_sum / i_sq_sum;
        float sin_corr = delta_theta;
        float cos_corr = (1 - delta_theta * delta_theta);
        double quad_misconf = asin(iq_sum / i_sq_sum);

        double deg_misconf = 360 * quad_misconf / (2 * M_PI);

        LOGD << "quad misconf = " << deg_misconf;
        results.phase_error = deg_misconf;
        {
            dim3 block_sz(16, 16);
            dim3 grid_sz((x_size + 15) / 16, (y_size + 15) / 16);
            ApplyPhaseCorrection<<<grid_sz, block_sz>>>(img.Data(), x_size, y_size, sin_corr, cos_corr);
        }
    }
}