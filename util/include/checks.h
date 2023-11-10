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