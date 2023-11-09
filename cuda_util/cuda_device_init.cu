/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/

#include "cuda_device_init.h"

#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>

#include "cuda_util.h"

namespace alus::cuda {
CudaInit::CudaInit() {
    init_future_ = std::async(std::launch::async, [this]() { this->QueryDevices(); });
}

bool CudaInit::IsFinished() const {
    if (!init_future_.valid()) {
        throw std::runtime_error("The future is already a past, invalid state queried.");
    }
    return init_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

void CudaInit::QueryDevices() {
    int device_count{};
    CHECK_CUDA_ERR(cudaGetDeviceCount(&device_count));
    if (!device_count) {
        throw std::runtime_error("No GPU devices detected");
    }
    for (int i{}; i < device_count; i++) {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERR(cudaGetDeviceProperties(&deviceProp, i));
        devices_.emplace_back(i, &deviceProp);
        // Whatever will first start invoking GPU, might be delayed if this thread does not finish.
        // But when waiting, a first invocation of GPU could be delayed by waiting here.
        // Also no error checking is done, because if there are errors, then sooner or later they will pop out
        // somewhere else.
        device_warmups_.emplace_back([i]() {
            cudaSetDevice(i);
            cudaFree(nullptr);
        });
    }
}

void CudaInit::CheckErrors() {
    if (!init_future_.valid()) {
        throw std::runtime_error("The future is already a past, invalid state queried.");
    }
    init_future_.get();
}

CudaInit::~CudaInit() {
    // Just in case wait if any left hanging.
    if (init_future_.valid()) {
        init_future_.wait_for(std::chrono::seconds(10));
    }

    for (auto& t : device_warmups_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

}  // namespace alus::cuda