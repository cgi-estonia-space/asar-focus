/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include <stdexcept>

#include <cufft.h>

#define CHECK_CUFFT_ERR(x) alus::cuda::CheckCuFFT(x, __FILE__, __LINE__)

namespace alus::cuda {

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

inline void CheckCufftSize(size_t workspace_size, cufftHandle plan) {
    size_t fft_workarea = 0;
    auto cufft_err = cufftGetSize(plan, &fft_workarea);
    if (cufft_err != CUFFT_SUCCESS || workspace_size < fft_workarea) {
        throw std::runtime_error("workspace size not enough for FFT plan");
    }
}

}
