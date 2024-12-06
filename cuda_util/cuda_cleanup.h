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

#include <cuda_runtime.h>
#include <cufft.h>
#include <memory>


// Wrappers around cuda APIs for exception safety
struct CudaMallocDeleter {
    void operator()(void *d_ptr) { cudaFree(d_ptr); }
};

template<class T>
using CudaMallocTypeCleanup = std::unique_ptr<T, CudaMallocDeleter>;

using CudaMallocCleanup = std::unique_ptr<void, CudaMallocDeleter>;

class CufftPlanCleanup {
public:
    explicit CufftPlanCleanup(cufftHandle plan) : plan_(plan) {}

    CufftPlanCleanup(const CufftPlanCleanup &) = delete;

    CufftPlanCleanup &operator=(const CufftPlanCleanup &) = delete;

    ~CufftPlanCleanup() { cufftDestroy(plan_); }

private:
    cufftHandle plan_;
};
