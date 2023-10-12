/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
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
