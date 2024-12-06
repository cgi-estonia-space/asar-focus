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
