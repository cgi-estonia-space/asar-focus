/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#include "fractional_doppler_centroid.cuh"

#include <sstream>

#include "alus_log.h"
#include "cuda_util/cuda_cleanup.h"
#include "util/checks.h"
#include "util/math_utils.h"

namespace {
constexpr int BLOCK_X_SIZE = 32;
constexpr int BLOCK_Y_SIZE = 32;
constexpr int TOTAL_BLOCK_SIZE = BLOCK_X_SIZE * BLOCK_Y_SIZE;
}  // namespace

__global__ void PhaseDifference(const cufftComplex* data, cufftComplex* result, int x_start, int x_end, int x_stride,
                                int y_size) {
    __shared__ cufftComplex shared_acc[TOTAL_BLOCK_SIZE];

    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int x = x_start + threadIdx.x + blockDim.x * blockIdx.x;
    const int shared_idx = threadIdx.x + threadIdx.y * BLOCK_X_SIZE;
    const int result_idx = blockIdx.x + blockIdx.y * gridDim.x;

    // each thread calculates a single phase difference between azimuth lines
    cufftComplex diff = {};
    if (x < x_end && y >= 1 && y < y_size) {
        const int prev_idx = (y - 1) * x_stride + x;
        const int cur_idx = y * x_stride + x;

        cufftComplex cur = data[cur_idx];
        cufftComplex prev = data[prev_idx];
        prev = cuConjf(prev);

        diff = cuCmulf(prev, cur);
    }

    // thread block shared memory reduced to single element
    shared_acc[shared_idx] = diff;
    __syncthreads();
    for (int s = TOTAL_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            cufftComplex t = cuCaddf(shared_acc[shared_idx], shared_acc[shared_idx + s]);
            shared_acc[shared_idx] = t;
        }
        __syncthreads();
    }

    if (shared_idx == 0) {
        result[result_idx] = shared_acc[0];
    }
}

// TODO(priit) replace with thrust::reduce?
__global__ void ReduceComplexSingleBlock(cufftComplex* d_array, int n_elem) {
    __shared__ cufftComplex shared_acc[TOTAL_BLOCK_SIZE];
    const int shared_idx = threadIdx.x;

    cufftComplex r = {};
    for (int idx = threadIdx.x; idx < n_elem; idx += TOTAL_BLOCK_SIZE) {
        if (idx < n_elem) {
            r = cuCaddf(r, d_array[idx]);
        }
    }
    shared_acc[shared_idx] = r;
    __syncthreads();

    for (int s = TOTAL_BLOCK_SIZE / 2; s > 0; s /= 2) {
        if (shared_idx < s) {
            cufftComplex t = cuCaddf(shared_acc[shared_idx], shared_acc[shared_idx + s]);
            shared_acc[shared_idx] = t;
        }
        __syncthreads();
    }

    if (shared_idx == 0) {
        d_array[0] = shared_acc[0];
    }
}

std::vector<double> CalculateDopplerCentroid(const DevicePaddedImage& d_img, double prf) {
    std::vector<double> dc_results;
    std::vector<double> dc_idx;

    constexpr int N_DC_CALC = 32;  // TODO investigate
    constexpr int POLY_ORDER = 2;

    const int azimuth_size = d_img.YSize();
    const int range_size = d_img.XSize();
    const int step = range_size / N_DC_CALC;

    for (int x_begin = 0; x_begin < range_size; x_begin += step) {
        const int x_end = x_begin + step;
        dim3 block_size(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        dim3 grid_size((step + block_size.x - 1) / block_size.x, (azimuth_size + block_size.y - 1) / block_size.y);

        const int total_blocks = grid_size.x * grid_size.y;
        size_t bsize = total_blocks * sizeof(cufftComplex);
        cufftComplex* d_sum;
        CHECK_CUDA_ERR(cudaMalloc(&d_sum, bsize));
        CudaMallocCleanup cleanup(d_sum);
        // Each block calculates a sum of 32x32 phase differences to d_sum, to avoid floating point error accumulation

        PhaseDifference<<<grid_size, block_size>>>(d_img.Data(), d_sum, x_begin, x_end, d_img.XStride(), azimuth_size);
        // Single thread block reducing d_sum, multistep reduction not needed
        ReduceComplexSingleBlock<<<1, 1024>>>(d_sum, total_blocks);

        // final result is at d_sum[0]
        std::complex<float> h_res;
        static_assert(sizeof(h_res) == sizeof(d_sum[0]));
        CHECK_CUDA_ERR(cudaMemcpy(&h_res, d_sum, sizeof(h_res), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaGetLastError());

        const double angle = atan2(h_res.imag(), h_res.real());
        const double fraction = angle / (2 * M_PI);
        const double doppler_centroid = prf * fraction;

        dc_results.push_back(doppler_centroid);
        dc_idx.push_back((x_begin + x_end) / 2);
    }

    auto dc_poly = Polyfit(dc_idx, dc_results, POLY_ORDER);

    LOGV << "Doppler Centroid result(range sample - Hz):";
    std::stringstream stream;
    char print_buf[50];
    for (int i = 0; i < N_DC_CALC; i++) {
        if (i && (i % 4) == 0) {
            LOGV << stream.str();
            stream.str("");
            stream.clear();
        }
        snprintf(print_buf, 50, "(%5d - %5.2f ) ", static_cast<int>(dc_idx.at(i)), dc_results.at(i));
        stream << print_buf;
    }
    LOGV << stream.str();
    stream.clear();
    LOGV << "Fitted polynomial";
    for (double e : dc_poly) {
        snprintf(print_buf, 50, "%g ", e);
        stream << print_buf;
    }
    LOGV << stream.str();
    return dc_poly;
}