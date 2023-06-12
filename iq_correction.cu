

#include "iq_correction.cuh"

#include "cuda_cleanup.h"

namespace {
    __global__ void ReduceSumRawIQData(const IQ8 *data, int x_size, int y_size, unsigned long long *result) {
        const int y_start = blockIdx.y;
        const int x_start = threadIdx.x;
        const int y_step = gridDim.y;
        const int x_step = blockDim.x;

        uint32_t n_samples = 0;
        uint32_t acc_i = 0;
        uint32_t acc_q = 0;
        for (int y = y_start; y < y_size; y += y_step) {
            for (int x = x_start; x < x_size; x += x_step) {
                int idx = (y * x_size) + x;
                auto sample = data[idx];

                acc_i += sample.i;
                acc_q += sample.q;
                n_samples++;
            }
        }

        // TODO shared reduction before atomics, if this should ever matter
        //  3 x SM count x 1024 atomicAdds on one memory location should not be too bad
        atomicAdd(result, acc_i);
        atomicAdd(result + 1, acc_q);
        atomicAdd(result + 2, n_samples);
    }

    __global__ void ApplyDCBiasKernel(const IQ8 *src, int src_x_size, cufftComplex *dest, int dest_x_size,
                                      int dest_x_stride, int dest_y_size, int dest_y_stride, float dc_i, float dc_q) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;

        const int dst_idx = y * dest_x_stride + x;
        const int src_idx = y * src_x_size + x;

        if (x < dest_x_size && y < dest_y_size) {
            // signal data index, convert from 2 x uint8_t to 2 x Float32
            auto iq8 = src[src_idx];
            dest[dst_idx].x = static_cast<float>(iq8.i) - dc_i;
            dest[dst_idx].y = static_cast<float>(iq8.q) - dc_q;
        } else if (x < dest_x_stride && y < dest_y_stride) {
            // zero pad index, both in range and azimuth
            dest[dst_idx] = {0.0f, 0.0f};
        }
    }


    __global__ void ReduceIQGain(const cufftComplex *data, int x_size, int x_stride, int y_size, double *result) {
        const int y_start = blockIdx.y;
        const int x_start = threadIdx.x;
        const int y_step = gridDim.y;
        const int x_step = blockDim.x;


        double i_sq = 0.0;
        double q_sq = 0.0;
        for (int y = y_start; y < y_size; y += y_step) {
            for (int x = x_start; x < x_size; x += x_step) {
                int idx = (y * x_stride) + x;
                auto sample = data[idx];
                float i = sample.x;
                float q = sample.y;

                i_sq += i*i;
                q_sq += q*q;

            }
        }

        // TODO shared reduction before atomics, if this should ever matter
        //  3 x SM count x 1024 atomicAdds on one memory location should not be too bad
        atomicAdd(result, i_sq);
        atomicAdd(result + 1, q_sq);
    }

    __global__ void ReduceQuadratureError(const cufftComplex *data, int x_size, int x_stride, int y_size, double *result) {
        const int y_start = blockIdx.y;
        const int x_start = threadIdx.x;
        const int y_step = gridDim.y;
        const int x_step = blockDim.x;


        double iq_mul = 0.0;
        double i_sq = 0.0;
        double q_sq = 0.0;
        for (int y = y_start; y < y_size; y += y_step) {
            for (int x = x_start; x < x_size; x += x_step) {
                int idx = (y * x_stride) + x;
                auto sample = data[idx];
                float i = sample.x;
                float q = sample.y;

                iq_mul += i*q;
                i_sq += i*i;
                q_sq += q*q;

            }
        }

        // TODO shared reduction before atomics, if this should ever matter
        //  3 x SM count x 1024 atomicAdds on one memory location should not be too bad
        atomicAdd(result, iq_mul);
        atomicAdd(result + 1, i_sq);
        atomicAdd(result + 2, q_sq);
    }

    __global__ void ApplyGain(cufftComplex *data, int x_size, int x_stride, int y_size, float i_gain) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;


        int idx = (y * x_stride) + x;

        data[idx].x /= i_gain;

    }

    __global__ void ApplyPhaseCorrection(cufftComplex *data, int x_size, int x_stride, int y_size, float sin_corr, float cos_corr) {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;


        int idx = (y * x_stride) + x;

        auto pix = data[idx];
        float i = pix.x;
        float q = pix.y;

        float corr_q = (q - i * sin_corr) / cos_corr;

        data[idx].y = corr_q;

    }
}  // namespace

#if 0
void FormatRawIQ8(const IQ8* d_signal_in, IQ8* d_signal_out, ImgFormat img, const std::vector<uint32_t>& offsets) {
    uint32_t* d_offsets;
    const size_t byte_size = offsets.size() * sizeof(offsets[0]);
    CHECK_CUDA_ERR(cudaMalloc(&d_offsets, byte_size));
    CudaMallocCleanup cleanup(d_offsets);
    CHECK_CUDA_ERR(cudaMemcpy(d_offsets, offsets.data(), byte_size, cudaMemcpyHostToDevice));
    dim3 block_size(16, 16);
    dim3 grid_size((img.range_size + 15) / 16, (img.azimuth_size + 15) / 16);

    static_assert(sizeof(IQ8) == 2);
    const int row_offset = img.data_line_offset / 2;  // accessed via IQ8*

    CorrectRawKernel<<<grid_size, block_size>>>(d_signal_in, d_signal_out, img.range_size, img.azimuth_size, row_offset,
                                                d_offsets);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}
#endif

void CalculateDCBias(const IQ8 *d_signal_data, int range_size, int azimuth_size, size_t multiprocessor_count, SARResults &result) {
    unsigned long long *d_result;
    const size_t res_bsize = 8 * 3;
    CHECK_CUDA_ERR(cudaMalloc(&d_result, res_bsize));
    CudaMallocCleanup cleanup(d_result);
    CHECK_CUDA_ERR(cudaMemset(d_result, 0, res_bsize));
    dim3 block_size(1024);
    dim3 grid_size(1, multiprocessor_count * 2);


    // Accumulate I and Q over the whole dataset
    ReduceSumRawIQData<<<grid_size, block_size>>>(d_signal_data, range_size, azimuth_size, d_result);

    unsigned long long h_result[3];
    CHECK_CUDA_ERR(cudaMemcpy(h_result, d_result, res_bsize, cudaMemcpyDeviceToHost));

    //const int total_samples = h_result[2];

    const int total_samples = range_size * azimuth_size;

    // find the average signal value for I and Q, should be ~15.5f for 5 bit data;
    result.dc_i = static_cast<double>(h_result[0]) / total_samples;
    result.dc_q = static_cast<double>(h_result[1]) / total_samples;
    printf("I/Q DC cuda = %f %f\n", result.dc_i, result.dc_q);
    printf("bias = %f %f\n", result.dc_i - 15.5, result.dc_q - 15.5);
    //result.total_samples = total_samples;

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

void ApplyDCBias(const IQ8 *d_signal_data, const SARResults &results, DevicePaddedImage &output) {
    dim3 block_size(16, 16);
    dim3 grid_size((output.XStride() + 15) / 16, (output.YStride() + 15) / 16);
    ApplyDCBiasKernel<<<grid_size, block_size>>>(d_signal_data, output.XSize(), output.Data(),
                                                 output.XSize(), output.XStride(), output.YSize(), output.YStride(),
                                                 results.dc_i, results.dc_q);

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

void CalculateAndApplyGain(DevicePaddedImage &img, SARResults &results, int multiprocessor_count) {

    double h_result[2];
    double* d_result;
    CHECK_CUDA_ERR(cudaMalloc(&d_result, sizeof(double) * 2));
    cudaMemset(d_result, 0, 16);
    CudaMallocCleanup clean(d_result);

    {
        dim3 block_size(1024);
        dim3 grid_size(1, multiprocessor_count * 2);
        ReduceIQGain<<<grid_size, block_size>>>(img.Data(), img.XSize(), img.XStride(), img.YSize(), d_result);
        CHECK_CUDA_ERR(cudaMemcpy(&h_result[0], d_result, sizeof(double)*2, cudaMemcpyDeviceToHost));

    }

    //printf("cuda gain results = %f %f\n", h_result[0], h_result[1]);
    double gain = sqrt(h_result[0]/h_result[1]);
    results.iq_gain = gain;
    printf("I/Q Gain imbalance = %f\n", gain);
    {
        dim3 block_size(32, 32);
        dim3 grid_size((img.XSize() + 31) / 32, (img.YSize() + 31) / 32);

        ApplyGain<<<grid_size, block_size>>>(img.Data(), img.XSize(), img.XStride(), img.YSize(), gain);
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());

}

void CalculateAndApplyPhaseCorrection(DevicePaddedImage &img, SARResults &results, int multiprocessor_count) {

    double h_result[3];
    double* d_result;
    CHECK_CUDA_ERR(cudaMalloc(&d_result, sizeof(double) * 3));
    cudaMemset(d_result, 0, 24);
    CudaMallocCleanup clean(d_result);

    {
        dim3 block_size(1024);
        dim3 grid_size(1, multiprocessor_count * 2);
        unsigned long long* d_cnt;
        cudaMalloc(&d_cnt, 8);
        cudaMemset(d_cnt, 0, 8);
        ReduceQuadratureError<<<grid_size, block_size>>>(img.Data(), img.XSize(), img.XStride(), img.YSize(), d_result);
        CHECK_CUDA_ERR(cudaMemcpy(&h_result[0], d_result, sizeof(double)*3, cudaMemcpyDeviceToHost));


    }


    double iq_mul = h_result[0];
    double i_sq = h_result[1];
    double q_sq = h_result[2];
    double c0 = iq_mul / i_sq;
    double c1 = iq_mul / q_sq;

    //printf("%f %f\n", c0, c1);

    double quad_misconf = asin(c0);

    //printf("quadruture depart: %f\n", quad_misconf);

    printf("quadrature depart = %f\n", (quad_misconf * 360.0) / (M_PI * 2));

    results.quad_depart = (quad_misconf * 360.0) / (M_PI * 2);

    float sin_corr = c0;
    float cos_corr = sqrt(1 - c0 * c0);


    //printf("cuda phase results = %f %f %f\n", h_result[0], h_result[1], h_result[2]);
    {
        dim3 block_size(32, 32);
        dim3 grid_size((img.XSize() + 31) / 32, (img.YSize() + 31) / 32);

        ApplyPhaseCorrection<<<grid_size, block_size>>>(img.Data(), img.XSize(), img.XStride(), img.YSize(), sin_corr, cos_corr);
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());

}


void iq_correct_cuda(const std::vector<IQ8> &vec, DevicePaddedImage &out) {

    printf("CUDA IQ CORRECTION:\n");
    IQ8* d_iq8;
    size_t byte_size = vec.size() * sizeof(IQ8);
    cudaMalloc(&d_iq8, byte_size);
    CudaMallocCleanup  clean(d_iq8);
    cudaMemcpy(d_iq8, vec.data(), byte_size, cudaMemcpyHostToDevice);

    SARResults results;
    CalculateDCBias(d_iq8, out.XSize(), out.YSize(), 32, results);
    ApplyDCBias(d_iq8, results, out);


    CalculateAndApplyGain(out, results, 32);

    CalculateAndApplyPhaseCorrection(out, results, 32);


    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());

}
