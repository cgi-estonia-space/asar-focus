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

#include "memory_policy.h"

namespace alus::cuda {

MemoryAllocationForecast::MemoryAllocationForecast(size_t alignment) : alignment_{alignment} {}

void MemoryAllocationForecast::Add(size_t bytes) {
    if (bytes % alignment_ != 0) {
        forecast_ += ((bytes / alignment_) + 1) * alignment_;
    } else {
        forecast_ += bytes;
    }
}

MemoryFitPolice::MemoryFitPolice(const CudaDevice& device, size_t percentage_allowed)
    : device_{device}, percentage_{percentage_allowed}, total_memory_{device_.GetTotalGlobalMemory()} {}

bool MemoryFitPolice::CanFit(size_t bytes) const {
    const auto allowed_memory =
        static_cast<size_t>(total_memory_ * (static_cast<double>(percentage_) / 100.0));
    if (bytes > allowed_memory) {
        return false;
    }

    if (device_.GetFreeGlobalMemory() < bytes) {
        return false;
    }

    return true;
}

}  // namespace alus::cuda
