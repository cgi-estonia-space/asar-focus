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
