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

#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "alus_log.h"

#define CHECK_CUDA_ERR(x) alus::checkCudaError(x, __FILE__, __LINE__)
#define REPORT_WHEN_CUDA_ERR(x) alus::ReportIfCudaError(x, __FILE__, __LINE__)
#define PRINT_DEVICE_FUNC_OCCUPANCY(dev_func, block_size) \
    LOGI << #dev_func << " occupancy " << alus::cuda::GetOccupancyPercentageFor(dev_func, block_size) << "%";

namespace alus {
namespace cuda {

int GetGridDim(int blockDim, int dataDim);

// An overload for calling "cudaSetDevice()" from CUDA API in host code.
cudaError_t CudaSetDevice(int device_nr);

// An overload for calling "cudaDeviceReset()" from CUDA API in host code.
cudaError_t CudaDeviceReset();

// An overload for calling "cudaDeviceSynchronize()" from CUDA API in host code.
cudaError_t CudaDeviceSynchronize();

}  // namespace cuda

class CudaErrorException final : public std::runtime_error {
public:
    CudaErrorException(cudaError_t cuda_error, const std::string& file, int line)
        : std::runtime_error("CUDA error (" + std::to_string(static_cast<int>(cuda_error)) + ")-'" +
                             std::string{cudaGetErrorString(cuda_error)} + "' at " + file + ":" +
                             std::to_string(line)) {}

    explicit CudaErrorException(const std::string& what) : std::runtime_error(std::move(what)) {}
};

inline void checkCudaError(cudaError_t const cudaErr, const char* file, int const line) {
    if (cudaErr != cudaSuccess) {
        throw alus::CudaErrorException(cudaErr, file, line);
    }
}

inline void ReportIfCudaError(const cudaError_t cuda_err, const char* file, const int line) {
    if (cuda_err != cudaSuccess) {
        LOGE << "CUDA error (" << cuda_err << ")-'" << cudaGetErrorString(cuda_err) << "' at " << file << ":" << line;
    }
}

}  // namespace alus
