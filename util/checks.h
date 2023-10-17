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

#include <cufft.h>

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

#define CHECK_CUFFT_ERR(x) CheckCuFFT(x, __FILE__, __LINE__)

// copy paste from cuda-11.2/samples/common/inc/helper_cuda.h
inline const char* CufftErrorStr(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}

inline void CheckCuFFT(cufftResult err, const char* file, int line) {
    if (err != CUFFT_SUCCESS) {
        std::string error_msg =
            std::string("cuFFT error = ") + CufftErrorStr(err) + " file = " + file + " line = " + std::to_string(line);
        throw std::runtime_error(error_msg);
    }
}

inline void CheckNullptr(void* ptr, const char* file, int line) {
    if (!ptr) {
        std::string error_msg = std::string("nullptr file = ") + file + " line = " + std::to_string(line);
        throw std::invalid_argument(error_msg);
    }
}

inline void CheckCufftSize(size_t workspace_size, cufftHandle plan) {
    size_t fft_workarea = 0;
    auto cufft_err = cufftGetSize(plan, &fft_workarea);
    if (cufft_err != CUFFT_SUCCESS || workspace_size < fft_workarea) {
        throw std::runtime_error("workspace size not enough for FFT plan");
    }
}

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