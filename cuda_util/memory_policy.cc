/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
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