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



#include "device_padded_image.cuh"


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


#include <numeric>
#include <vector>

#include <cuComplex.h>

#include "cuda_cleanup.h"

namespace {
    __global__ void ZeroFillPaddingsKernel(cufftComplex* data, int x_size, int y_size, int x_stride, int y_stride) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;

        const int idx = x + y * x_stride;

        if ((x >= x_size && x < x_stride) && (y >= y_size && y < y_stride)) {
            data[idx] = {};
        }
    }

    __global__ void MultiplyDataKernel(cufftComplex* data, int x_size, int y_size, int x_stride, float multiplier) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;

        const int idx = x + y * x_stride;

        if (x < x_size && y < y_size) {
            data[idx].x *= multiplier;
            data[idx].y *= multiplier;
        }
    }

    __global__ void NaNZeroKernel(cufftComplex* data, int x_size, int y_size) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;

        const int idx = x + y * x_size;

        if (x < x_size && y < y_size) {
            if(std::isnan(data[idx].x))
            {
                data[idx].x = 0.0f;
            }
            if(std::isnan(data[idx].y))
            {
                data[idx].y = 0.0f;
            }
        }
    }

// original code from with modification to use cufftComplex instead
// https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
    constexpr int TRANSPOSE_BLOCK_DIM = 16;
    __global__ void TransposeKernel(cufftComplex* idata, cufftComplex* odata, int width, int height) {
        __shared__ cufftComplex block[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM + 1];

        // read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
        int xIndex = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.x;
        int yIndex = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.y;
        if ((xIndex < width) && (yIndex < height)) {
            unsigned int index_in = yIndex * width + xIndex;
            block[threadIdx.y][threadIdx.x] = idata[index_in];
        }

        // synchronise to ensure all writes to block[][] have completed
        __syncthreads();

        // write the transposed matrix tile to global memory (odata) in linear order
        xIndex = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.x;
        yIndex = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.y;
        if ((xIndex < height) && (yIndex < width)) {
            int index_out = yIndex * height + xIndex;
            odata[index_out] = block[threadIdx.x][threadIdx.y];
        }
    }

    constexpr int REDUCE_BLOCK_SIZE = 1024;

    __global__ void ReduceIntensity(const cufftComplex* data, int x_size, int y_size, int x_stride, float* result) {
        __shared__ float shared_acc[REDUCE_BLOCK_SIZE];
        const int y_start = blockIdx.y;
        const int x_start = threadIdx.x;
        const int shared_idx = x_start;
        const int result_idx = blockIdx.y;
        const int y_step = gridDim.y;
        const int x_step = blockDim.x;

        double acc = {};
        for (int y = y_start; y < y_size; y += y_step) {
            float inner_acc = {};
            for (int x = x_start; x < x_size; x += x_step) {
                int idx = (y * x_stride) + x;
                auto sample = data[idx];
                const float i = sample.x;
                const float q = sample.y;
                inner_acc += i * i + q * q;
            }
            acc += inner_acc;
        }

        shared_acc[shared_idx] = acc;
        __syncthreads();

        for (int s = REDUCE_BLOCK_SIZE / 2; s > 0; s /= 2) {
            if (shared_idx < s) {
                float t = shared_acc[shared_idx] + shared_acc[shared_idx + s];
                shared_acc[shared_idx] = t;
            }
            __syncthreads();
        }

        if (shared_idx == 0) {
            result[result_idx] = shared_acc[0];
        }
    }


    __global__ void ReduceStdDevDiffSquares(const cufftComplex* src, int x_size, int y_size, int x_stride, float i_mean, float q_mean, float* i_result_arr,
                                    float* q_result_arr) {
        __shared__ float i_shared_acc[REDUCE_BLOCK_SIZE];
        __shared__ float q_shared_acc[REDUCE_BLOCK_SIZE];

        const int shared_idx = threadIdx.x;
        const int result_idx = blockIdx.y;
        const int y_start = blockIdx.y;
        const int x_start = threadIdx.x;
        const int y_step = gridDim.y;
        const int x_step = blockDim.x;

        double i_diff_sq_sum = 0.0;
        double q_diff_sq_sum = 0.0;
        for (int y = y_start; y < y_size; y += y_step) {
            for (int x = x_start; x < x_size; x += x_step) {
                int idx = (y * x_stride) + x;
                auto sample = src[idx];
                float i = sample.x;
                float q = sample.y;

                float diff_i = i - i_mean;
                float diff_q = q - q_mean;

                if (!std::isnan(i) && !std::isnan(q)) {
                    i_diff_sq_sum += diff_i * diff_i;
                    q_diff_sq_sum += diff_q * diff_q;
                }
            }
        }

        // store each thread's result in shared memory, ensure all threads reach this point
        i_shared_acc[shared_idx] = i_diff_sq_sum;
        q_shared_acc[shared_idx] = q_diff_sq_sum;
        __syncthreads();

        // reduce the shared memory array to the first element
        // Reduction #3: Sequential Addressing from NVidia Optimizing Parallel Reduction
        for (int s = REDUCE_BLOCK_SIZE / 2; s > 0; s /= 2) {
            if (shared_idx < s) {
                i_shared_acc[shared_idx] += i_shared_acc[shared_idx + s];
                q_shared_acc[shared_idx] += q_shared_acc[shared_idx + s];
            }
            __syncthreads();
        }

        // write the final result to global memory
        if (shared_idx == 0) {
            i_result_arr[result_idx] = i_shared_acc[0];
            q_result_arr[result_idx] = q_shared_acc[0];
        }
    }

    __global__ void ReduceMean(const cufftComplex* src, int x_size, int y_size, int x_stride, float* i_result_arr,
                                            float* q_result_arr) {
        __shared__ float i_shared_acc[REDUCE_BLOCK_SIZE];
        __shared__ float q_shared_acc[REDUCE_BLOCK_SIZE];

        const int shared_idx = threadIdx.x;
        const int result_idx = blockIdx.y;
        const int y_start = blockIdx.y;
        const int x_start = threadIdx.x;
        const int y_step = gridDim.y;
        const int x_step = blockDim.x;

        double i_sum = 0.0;
        double q_sum = 0.0;
        for (int y = y_start; y < y_size; y += y_step) {
            for (int x = x_start; x < x_size; x += x_step) {
                int idx = (y * x_stride) + x;
                auto sample = src[idx];
                float i = sample.x;
                float q = sample.y;

                if (!std::isnan(i) && !std::isnan(q)) {
                    i_sum += i;
                    q_sum += q;
                }
            }
        }

        // store each thread's result in shared memory, ensure all threads reach this point
        i_shared_acc[shared_idx] = i_sum;
        q_shared_acc[shared_idx] = q_sum;
        __syncthreads();

        // reduce the shared memory array to the first element
        // Reduction #3: Sequential Addressing from NVidia Optimizing Parallel Reduction
        for (int s = REDUCE_BLOCK_SIZE / 2; s > 0; s /= 2) {
            if (shared_idx < s) {
                i_shared_acc[shared_idx] += i_shared_acc[shared_idx + s];
                q_shared_acc[shared_idx] += q_shared_acc[shared_idx + s];
            }
            __syncthreads();
        }

        // write the final result to global memory
        if (shared_idx == 0) {
            i_result_arr[result_idx] = i_shared_acc[0];
            q_result_arr[result_idx] = q_shared_acc[0];
        }
    }


}  // namespace


    void DevicePaddedImage::MultiplyData(float multiplier) {
        dim3 block_size(16, 16);
        dim3 grid_size((x_stride_ + 15) / 16, (y_stride_ + 15) / 16);
        MultiplyDataKernel<<<grid_size, block_size>>>(d_data_, x_size_, y_size_, x_stride_, multiplier);
    }
    void DevicePaddedImage::ZeroFillPaddings() {
        dim3 block_size(16, 16);
        dim3 grid_size((x_stride_ + 15) / 16, (y_stride_ + 15) / 16);
        ZeroFillPaddingsKernel<<<grid_size, block_size>>>(d_data_, x_size_, y_size_, x_stride_, y_stride_);
    }

    void DevicePaddedImage::Transpose() {
        // TODO(priit) investigate inplace rectangular transpose?
        cufftComplex* d_transposed;
        CHECK_CUDA_ERR(cudaMalloc(&d_transposed, TotalByteSize()));

        dim3 block_size(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_size((x_stride_ + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                       (y_stride_ + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        TransposeKernel<<<grid_size, block_size>>>(d_data_, d_transposed, x_stride_, y_stride_);
        cudaFree(d_data_);
        std::swap(x_stride_, y_stride_);
        std::swap(x_size_, y_size_);
        d_data_ = d_transposed;
    }

    void DevicePaddedImage::Transpose(CudaWorkspace& d_workspace) {
        VerifyExtMemorySize(TotalByteSize(), d_workspace.ByteSize());

        dim3 block_size(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
        dim3 grid_size((x_stride_ + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM,
                       (y_stride_ + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM);

        TransposeKernel<<<grid_size, block_size>>>(d_data_, d_workspace.GetAs<cufftComplex>(), x_stride_, y_stride_);
        std::swap(x_stride_, y_stride_);
        std::swap(x_size_, y_size_);

        // swap pointers, so transposed image is owned by this object, and the old memory is now usable by others
        auto* d_workspace_ptr = static_cast<cufftComplex*>(d_workspace.ReleaseMemory());
        d_workspace.Reset(d_data_, TotalByteSize());
        std::swap(d_workspace_ptr, d_data_);
    }

    double DevicePaddedImage::CalcTotalIntensity(size_t sm_count) {
        size_t n_blocks = sm_count;
        dim3 grid_dim(1, n_blocks);
        float* d_block_sums;
        size_t byte_size = n_blocks * sizeof(float);
        CHECK_CUDA_ERR(cudaMalloc(&d_block_sums, byte_size));
        CudaMallocCleanup  clean(d_block_sums);

        ReduceIntensity<<<grid_dim, REDUCE_BLOCK_SIZE>>>(d_data_, x_size_, y_size_, x_stride_, d_block_sums);
        std::vector<float> h_block_sums(n_blocks);
        CHECK_CUDA_ERR(cudaMemcpy(h_block_sums.data(), d_block_sums, byte_size, cudaMemcpyDeviceToHost));

        return std::accumulate(h_block_sums.begin(), h_block_sums.end(), 0.0);
    }

    void DevicePaddedImage::ZeroNaNs()
    {
        dim3 block_sz(16, 16);
        dim3 grid_sz((x_stride_ + 15) / 16, (y_stride_ + 15) / 16);

        NaNZeroKernel<<<grid_sz, block_sz>>>(d_data_, x_stride_, y_stride_);

    }

    cufftComplex DevicePaddedImage::CalcStdDev(float i_mean, float q_mean, size_t total_samples) const
    {
        size_t n_blocks = 50;
        size_t byte_size = n_blocks * sizeof(float);


        dim3 grid_dim(1, n_blocks);

        float* i_diff_sum;
        float* q_diff_sum;
        CHECK_CUDA_ERR(cudaMalloc(&i_diff_sum, byte_size));
        CudaMallocCleanup clean_i(i_diff_sum);

        CHECK_CUDA_ERR(cudaMalloc(&q_diff_sum, byte_size));
        CudaMallocCleanup clean_q(q_diff_sum);

        ReduceStdDevDiffSquares<<<grid_dim, REDUCE_BLOCK_SIZE>>>(d_data_, x_size_, y_size_, x_stride_, i_mean, q_mean, i_diff_sum, q_diff_sum);

        // each thread block has summed it's part of squared mean differences


        // do the second part of reduction on cpu, beacuse this is not a speed bottleneck in the project
        std::vector<float> h_i_sum(n_blocks);
        CHECK_CUDA_ERR(cudaMemcpy(h_i_sum.data(), i_diff_sum, byte_size, cudaMemcpyDeviceToHost));
        std::vector<float> h_q_sum(n_blocks);
        CHECK_CUDA_ERR(cudaMemcpy(h_q_sum.data(), q_diff_sum, byte_size, cudaMemcpyDeviceToHost));

        cufftComplex r = {};
        r.x = std::accumulate(h_i_sum.begin(), h_i_sum.end(), 0.0);
        r.y = std::accumulate(h_q_sum.begin(), h_q_sum.end(), 0.0);

        if (total_samples == 0) {
            total_samples = x_size_ * y_size_;
        }
        // variance
        r.x /= total_samples;
        r.y /= total_samples;

        // standard deviation
        r.x = sqrt(r.x);
        r.y = sqrt(r.y);

        return r;
    }

    cufftComplex DevicePaddedImage::CalcMean(size_t total_samples) const
    {
        size_t n_blocks = 50;
        size_t byte_size = n_blocks * sizeof(float);

        dim3 grid_dim(1, n_blocks);

        float* i_sum;
        float* q_sum;
        CHECK_CUDA_ERR(cudaMalloc(&i_sum, byte_size));
        CudaMallocCleanup clean_i(i_sum);

        CHECK_CUDA_ERR(cudaMalloc(&q_sum, byte_size));
        CudaMallocCleanup clean_q(q_sum);

        ReduceMean<<<grid_dim, REDUCE_BLOCK_SIZE>>>(d_data_, x_size_, y_size_, x_stride_, i_sum, q_sum);

        // each thread block has summed it's part of I & Q channels


        // do the second part of reduction on cpu, beacuse this is not a speed bottleneck in the project
        std::vector<float> h_i_sum(n_blocks);
        CHECK_CUDA_ERR(cudaMemcpy(h_i_sum.data(), i_sum, byte_size, cudaMemcpyDeviceToHost));
        std::vector<float> h_q_sum(n_blocks);
        CHECK_CUDA_ERR(cudaMemcpy(h_q_sum.data(), q_sum, byte_size, cudaMemcpyDeviceToHost));

        cufftComplex r = {};
        r.x = std::accumulate(h_i_sum.begin(), h_i_sum.end(), 0.0);
        r.y = std::accumulate(h_q_sum.begin(), h_q_sum.end(), 0.0);

        if (total_samples == 0) {
            total_samples = x_size_ * y_size_;
        }

        r.x /= total_samples;
        r.y /= total_samples;

        return r;
    }