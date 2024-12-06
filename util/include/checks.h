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

#include <stdexcept>

#include "alus_log.h"

#ifndef __NVCC__
#include <gdal/gdal_priv.h>
#define CHECK_GDAL_ERROR(err) CheckGdalError(err, __FILE__, __LINE__)

class GdalErrorException final : public std::runtime_error {
public:
    GdalErrorException(CPLErr const errType, CPLErrorNum const errNum, std::string_view errMsg, std::string_view src,
                       int srcLine)
        : std::runtime_error("GDAL error no " + std::to_string(static_cast<int>(errNum)) + " type " +
                             std::to_string(static_cast<int>(errType)) + " - '" + std::string{errMsg} + "' at " +
                             std::string{src} + ":" + std::to_string(srcLine)),
          gdal_error_{errNum},
          file_{src},
          line_{srcLine} {}

    [[nodiscard]] CPLErrorNum GetGdalError() const { return gdal_error_; }
    [[nodiscard]] std::string_view GetSource() const { return file_; }
    [[nodiscard]] int GetLine() const { return line_; }

private:
    CPLErrorNum const gdal_error_;
    std::string file_;
    int const line_;
};

inline void CheckGdalError(CPLErr const err, char const* file, int const line) {
    if (err != CE_None) {
        throw GdalErrorException(err, CPLGetLastErrorNo(), CPLGetLastErrorMsg(), file, line);
    }
}
#endif

#define CHECK_BOOL(b) VerifyBool(b, __FILE__, __LINE__)

inline void VerifyBool(bool cond, const char* file, int line) {
    if (!cond) {
        LOGE << "error  file = " << file << " line = " << line;
        exit(1);
    }
}

#define ERROR_EXIT(msg) ErrorExit(msg, __FILE__, __LINE__)

[[noreturn]] inline void ErrorExit(std::string msg, const char* file, int line) {
    LOGE << "error msg = " << msg << " file = " << file << " line = " << line;
    exit(1);
}