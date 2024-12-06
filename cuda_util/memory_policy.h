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

#include "cuda_device.h"

namespace alus::cuda {

class MemoryAllocationForecast final {
public:
    MemoryAllocationForecast() = delete;
    explicit MemoryAllocationForecast(size_t alignment);

    void Add(size_t bytes);
    [[nodiscard]] size_t Get() const { return forecast_; }

private:
    size_t alignment_;
    size_t forecast_{};
};

class MemoryFitPolice final {
public:
    MemoryFitPolice() = delete;
    MemoryFitPolice(const CudaDevice& device, size_t percentage_allowed);

    [[nodiscard]] bool CanFit(size_t bytes) const;

private:
    const CudaDevice& device_;
    const size_t percentage_;
    const size_t total_memory_;
};

}  // namespace alus::cuda
