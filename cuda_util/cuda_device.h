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

#include <cstddef>
#include <string>
#include <string_view>

namespace alus::cuda {
class CudaDevice final {
public:
    CudaDevice() = delete;
    /*
     * Parses GPU device properties from 'cudaDeviceProp' struct pointer.
     * It is opaque one here in order to not include CUDA SDK headers to host compilation.
     */
    CudaDevice(int device_nr, void* device_prop);

    void Set() const;

    [[nodiscard]] int GetDeviceNr() const { return device_nr_; }
    [[nodiscard]] std::string_view GetName() const { return name_; }
    [[nodiscard]] size_t GetCcMajor() const { return cc_major_; }
    [[nodiscard]] size_t GetCcMinor() const { return cc_minor_; }
    [[nodiscard]] size_t GetSmCount() const { return sm_count_; }
    [[nodiscard]] size_t GetMaxThreadsPerSm() const { return max_threads_per_sm_; }
    [[nodiscard]] size_t GetWarpSize() const { return warp_size_; }
    [[nodiscard]] size_t GetTotalGlobalMemory() const { return total_global_memory_; };
    [[nodiscard]] size_t GetFreeGlobalMemory() const;
    [[nodiscard]] size_t GetMemoryAlignment() const { return alignment_; }

private:
    int device_nr_;
    size_t cc_major_;
    size_t cc_minor_;
    std::string name_;
    size_t sm_count_;
    size_t max_threads_per_sm_;
    size_t warp_size_;
    size_t total_global_memory_;
    size_t alignment_;
};
}  // namespace alus::cuda
