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