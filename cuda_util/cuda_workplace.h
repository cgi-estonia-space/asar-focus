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

#include <cstdlib>

#include "cuda_util.h"


// Can be thought as a std::unique_ptr + buffer byte size
class CudaWorkspace {
    void *d_ptr_ = nullptr;
    size_t byte_size_ = 0;

public:
    CudaWorkspace() = default;

    explicit CudaWorkspace(size_t byte_size) {
        CHECK_CUDA_ERR(cudaMalloc(&d_ptr_, byte_size));
        byte_size_ = byte_size;
    }

    CudaWorkspace(CudaWorkspace &&oth) noexcept: d_ptr_(oth.d_ptr_), byte_size_(oth.byte_size_) {
        oth.d_ptr_ = nullptr;
        oth.byte_size_ = 0;
    }

    CudaWorkspace &operator=(CudaWorkspace &&oth) noexcept {
        if (d_ptr_) {
            cudaFree(d_ptr_);
        }
        d_ptr_ = oth.d_ptr_;
        byte_size_ = oth.byte_size_;
        oth.d_ptr_ = nullptr;
        oth.byte_size_ = 0;
        return *this;
    }

    template<class T>
    [[nodiscard]] T *GetAs() {
        return static_cast<T *>(d_ptr_);
    }

    template<class T>
    [[nodiscard]] const T *GetAs() const {
        return static_cast<const T *>(d_ptr_);
    }

    [[nodiscard]] void *Get() { return d_ptr_; }

    [[nodiscard]] const void *Get() const { return d_ptr_; }

    [[nodiscard]] size_t ByteSize() const { return byte_size_; }

    [[nodiscard]] void *ReleaseMemory() {
        void *ret = d_ptr_;
        byte_size_ = 0;
        d_ptr_ = nullptr;
        return ret;
    }

    void Reset(void *d_ptr, size_t byte_size) {
        if (d_ptr_) {
            CHECK_CUDA_ERR(cudaFree(d_ptr_));
        }
        d_ptr_ = d_ptr;
        byte_size_ = byte_size;
    }

    void Reset() { Reset(nullptr, 0U); }

    ~CudaWorkspace() { cudaFree(d_ptr_); }
};
