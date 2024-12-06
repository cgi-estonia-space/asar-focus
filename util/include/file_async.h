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

#include <chrono>
#include <cstddef>
#include <fstream>
#include <future>
#include <string_view>

namespace alus::util {

class FileAsync {
public:
    FileAsync() = delete;
    FileAsync(std::string_view path);

    FileAsync(const FileAsync&) = delete;
    FileAsync& operator=(const FileAsync&) = delete;

    bool CanWrite(const std::chrono::milliseconds& timeout = std::chrono::milliseconds(0));
    void Write(const void* data, size_t bytes);
    void Write(size_t position, const void* data, size_t bytes);
    void WriteSync(const void* data, size_t bytes);

    ~FileAsync();

private:
    void ThrowOnErrors();
    void Start();

    bool has_been_opened_{false};
    std::ofstream file_handle_;
    std::future<void> async_handle_;
    const std::string filename_path_;
};

}  // namespace alus::util