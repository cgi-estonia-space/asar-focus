/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "cuda_device.h"

#include <cuda_runtime_api.h>

#include "cuda_util.h"

namespace alus::cuda {

CudaDevice::CudaDevice(int device_nr, void* device_prop) : device_nr_{device_nr} {
    cudaDeviceProp* dev = reinterpret_cast<cudaDeviceProp*>(device_prop);
    cc_major_ = dev->major;
    cc_minor_ = dev->minor;
    name_ = dev->name;
    sm_count_ = dev->multiProcessorCount;
    max_threads_per_sm_ = dev->maxThreadsPerMultiProcessor;
    warp_size_ = dev->warpSize;
    total_global_memory_ = dev->totalGlobalMem;
    alignment_ = dev->textureAlignment;
}

void CudaDevice::Set() const {
    CHECK_CUDA_ERR(cudaSetDevice(device_nr_));
}

size_t CudaDevice::GetFreeGlobalMemory() const {
    Set();
    size_t total;
    size_t free;
    CHECK_CUDA_ERR(cudaMemGetInfo(&free, &total));

    return free;
}

}  // namespace alus::cuda
