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

#include <future>
#include <thread>
#include <vector>

#include "cuda_device.h"

namespace alus::cuda {
class CudaInit final {
public:
    CudaInit();

    [[nodiscard]] bool IsFinished() const;
    void CheckErrors();

    [[nodiscard]] const std::vector<CudaDevice>& GetDevices() const { return devices_; }

    ~CudaInit();

private:
    void QueryDevices();

    std::vector<CudaDevice> devices_;
    std::future<void> init_future_;
    std::vector<std::thread> device_warmups_;
};
}  // namespace alus::cuda