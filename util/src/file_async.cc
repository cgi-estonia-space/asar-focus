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

#include "file_async.h"

#include <ios>
#include <stdexcept>

namespace alus::util {

FileAsync::FileAsync(std::string_view path) : filename_path_{path} {
    async_handle_ = std::async(std::launch::async, [this]() { this->Start(); });
}

bool FileAsync::CanWrite(const std::chrono::milliseconds& timeout) {
    if (!async_handle_.valid()) {
        return !file_handle_.fail() && has_been_opened_;
    }

    const auto status = async_handle_.wait_for(timeout);
    if (status == std::future_status::ready) {
        async_handle_.get();  // Check for exceptions
        return !file_handle_.fail() && has_been_opened_;
    }

    return false;
}

void FileAsync::Write(const void* data, size_t bytes) {
    ThrowOnErrors();
    async_handle_ = std::async(std::launch::async, [this, data, bytes]() { this->WriteSync(data, bytes); });
}

void FileAsync::Write(size_t position, const void* data, size_t bytes) {
    ThrowOnErrors();
    file_handle_.seekp(position, std::ios::beg);
    Write(data, bytes);
}

void FileAsync::Start() {
    file_handle_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    file_handle_.open(filename_path_.c_str(), std::ios::out | std::ios::binary);
    if (file_handle_.good() && file_handle_.is_open()) {
        has_been_opened_ = true;
    }
}

void FileAsync::WriteSync(const void* data, size_t bytes) {
    file_handle_.write(reinterpret_cast<const char*>(data), bytes);
}

void FileAsync::ThrowOnErrors() {
    if (!CanWrite()) {
        throw std::runtime_error(
            "FileAsync operation was called when either file stream has failed or another write operation is ongoing.");
    }
}

FileAsync::~FileAsync() {
    // Just in case wait if any left hanging.
    if (async_handle_.valid()) {
        async_handle_.wait_for(std::chrono::seconds(10));
    }

    if (file_handle_.is_open()) {
        try {
            file_handle_.close();
        } catch (const std::ios_base::failure&) {
            // Well yeah, nice, but cannot do anything anymore.
        }
    }
}

}  // namespace alus::util