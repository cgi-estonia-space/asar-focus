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
#include "ers_env_format.h"

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

__global__ void ConvertErsImSamplesToComplexKernel(uint8_t* samples, int samples_range_size_bytes, int packets,
                                                   uint16_t* swst_codes, uint16_t min_swst, cufftComplex* target_buf,
                                                   int target_range_width) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int x_size = samples_range_size_bytes;

    if (x < x_size && y < packets) {
        const int buffer_index = y * x_size + x;
        const int target_buffer_index = y * target_range_width + (x / 2);
        // ERS SWST multiplier = 4
        int offset = 4 * (swst_codes[y] - min_swst);
        if (x % 2 == 0) {
            target_buf[target_buffer_index + offset].x = static_cast<float>(samples[buffer_index]);
        } else {
            target_buf[target_buffer_index + offset].y = static_cast<float>(samples[buffer_index]);
        }
    }
}

void ConvertErsImSamplesToComplex(uint8_t* samples, size_t source_range_samples, size_t packets, uint16_t* swst_codes,
                                  uint16_t min_swst, cufftComplex* gpu_buffer, size_t target_range_padded_samples) {
    const auto samples_range_bytes = source_range_samples * 2 * sizeof(uint8_t);
    const auto samples_total_bytes = packets * samples_range_bytes;
    uint8_t* samples_gpu;
    CHECK_CUDA_ERR(cudaMalloc(&samples_gpu, samples_total_bytes));
    CHECK_CUDA_ERR(cudaMemcpy(samples_gpu, samples, samples_total_bytes, cudaMemcpyHostToDevice));
    uint16_t* swst_codes_gpu;
    CHECK_CUDA_ERR(cudaMalloc(&swst_codes_gpu, packets * sizeof(uint16_t)));
    CHECK_CUDA_ERR(cudaMemcpy(swst_codes_gpu, swst_codes, packets * sizeof(uint16_t), cudaMemcpyHostToDevice));

    const auto x_size = ers::highrate::MEASUREMENT_DATA_SIZE_BYTES;
    const auto y_size = packets;
    dim3 block_size(16, 16);
    dim3 grid_sz((x_size + 15) / 16, (y_size + 15) / 16);
    ConvertErsImSamplesToComplexKernel<<<grid_sz, block_size>>>(
        samples_gpu, samples_range_bytes, packets, swst_codes_gpu, min_swst, gpu_buffer, target_range_padded_samples);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

}  // namespace alus::asar::envformat