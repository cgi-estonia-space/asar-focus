/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "envisat_format_kernels.h"

#include "cuda_util/cuda_algorithm.cuh"
#include "cuda_util/cuda_bit.cuh"
#include "envisat_types.h"

namespace alus::asar::envformat {

__global__ void CalibrateClampToBigEndian(cufftComplex* data, int x_size_stride, int x_size, int y_size,
                                          char* dest_space, int record_header_size, float calibration_constant) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int src_idx = (y * x_size_stride) + x;
    const int mds_x_size = x_size * sizeof(IQ16) + record_header_size;

    if (x < x_size && y < y_size) {
        auto pix = data[src_idx];
        IQ16 result;
        result.i = static_cast<int16_t>(
            alus::cuda::algorithm::clamp<float>(pix.x * calibration_constant, INT16_MIN, INT16_MAX));
        result.q = static_cast<int16_t>(
            alus::cuda::algorithm::clamp<float>(pix.y * calibration_constant, INT16_MIN, INT16_MAX));

        result.i = alus::cuda::bit::Byteswap(result.i);
        result.q = alus::cuda::bit::Byteswap(result.q);

        // |HEADER...|packet data....................................|
        // |HEADER...|packet data....................................|
        // ...
        const int dest_idx = (y * mds_x_size) + record_header_size + (x * sizeof(IQ16));
        if (x == 0) {
            for (int i{record_header_size}; i >= 0; i--) {
                dest_space[dest_idx - i] = 0x00;
            }
        }

        dest_space[dest_idx] = static_cast<char>(result.i & 0x00FF);
        dest_space[dest_idx + 1] = static_cast<char>((result.i >> 8) & 0x00FF);
        dest_space[dest_idx + 2] = static_cast<char>(result.q & 0x00FF);
        dest_space[dest_idx + 3] = static_cast<char>((result.q >> 8) & 0x00FF);
    }
}

void ConditionResults(DevicePaddedImage& img, char* dest_space, size_t record_header_size, float calibration_constant) {
    const auto x_size_stride = img.XStride();  // Need to count for FFT padding when rows are concerned
    const auto x_size = img.XSize();
    const auto y_size = img.YSize();  // No need to calculate on Y padded FFT data
    dim3 block_sz(16, 16);
    dim3 grid_sz((x_size_stride + 15) / 16, (y_size + 15) / 16);
    CalibrateClampToBigEndian<<<grid_sz, block_sz>>>(img.Data(), x_size_stride, x_size, y_size, dest_space,
                                                     record_header_size, calibration_constant);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

}  // namespace alus::asar::envformat