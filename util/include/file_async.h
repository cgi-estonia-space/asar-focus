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