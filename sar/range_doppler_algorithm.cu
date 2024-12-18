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
#include "range_doppler_algorithm.cuh"

#include "checks.h"
#include "cuda_cleanup.h"
#include "cufft_checks.h"
#include "cufft_plan.h"
#include "focussing_details.h"

#define INPLACE_AZIMUTH_FFT 0  // 0 -> transpose + range FFT + transpose, 1 -> azimuth FFT

// 0 -> Nearest neighbor / no interpolator
//  -> 8 point windowed sinc interpolator
#define RCMC_INTERPOLATOR_SIZE 16

#if defined(RCMC_INTERPOLATOR_SIZE) && RCMC_INTERPOLATOR_SIZE == 0

#elif RCMC_INTERPOLATOR_SIZE == 8

// all windows generated with NumPy 1.17.4
// No window
//__constant__ const float WINDOW[8] = {1, 1, 1, 1, 1, 1, 1};

// numpy.blackman(8)
__constant__ const float WINDOW[8] = {-1.38777878e-17, 9.04534244e-02, 4.59182958e-01, 9.20363618e-01,
                                      9.20363618e-01,  4.59182958e-01, 9.04534244e-02, -1.38777878e-17};
// numpy.hamming(8)
//__constant__ const float WINDOW[8] = {0.08,       0.25319469, 0.64235963, 0.95444568,
//                                      0.95444568, 0.64235963, 0.25319469, 0.08};

#elif RCMC_INTERPOLATOR_SIZE == 16
//__constant__ const float WINDOW[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
// numpy.blackman(16)
__constant__ const float WINDOW[16] = {-1.38777878e-17, 1.67577197e-02, 7.70724198e-02, 2.00770143e-01,
                                       3.94012424e-01,  6.30000000e-01, 8.49229857e-01, 9.82157437e-01,
                                       9.82157437e-01,  8.49229857e-01, 6.30000000e-01, 3.94012424e-01,
                                       2.00770143e-01,  7.70724198e-02, 1.67577197e-02, -1.38777878e-17};

// numpy.hamming(16)
//__constant__ const float WINDOW[16] = {0.08,       0.11976909, 0.23219992, 0.39785218, 0.58808309, 0.77,
//                                       0.91214782, 0.9899479,  0.9899479,  0.91214782, 0.77,       0.58808309,
//                                       0.39785218, 0.23219992, 0.11976909, 0.08};
#elif RCMC_INTERPOLATOR_SIZE == 32
// numpy.blackman(32)
__constant__ const float WINDOW[32] = {
    -1.38777878e-17, 3.75165430e-03, 1.56384477e-02, 3.74026996e-02, 7.14646070e-02, 1.20286463e-01, 1.85646724e-01,
    2.67954971e-01,  3.65735039e-01, 4.75378537e-01, 5.91228597e-01, 7.06000789e-01, 8.11493284e-01, 8.99490429e-01,
    9.62730703e-01,  9.95797057e-01, 9.95797057e-01, 9.62730703e-01, 8.99490429e-01, 8.11493284e-01, 7.06000789e-01,
    5.91228597e-01,  4.75378537e-01, 3.65735039e-01, 2.67954971e-01, 1.85646724e-01, 1.20286463e-01, 7.14646070e-02,
    3.74026996e-02,  1.56384477e-02, 3.75165430e-03, -1.38777878e-17};
#else
#error "Unsupported RCMC interpolator size"
#endif

#if RCMC_INTERPOLATOR_SIZE > 0
constexpr int N_INTERPOLATOR = RCMC_INTERPOLATOR_SIZE;
static_assert(sizeof(WINDOW) / sizeof(WINDOW[0]) == N_INTERPOLATOR);

__device__ float sinc(float x) {
    const float pi = M_PI;
    if (x == 0) return 1.0f;
    return sinf(pi * x) / (pi * x);
}
#endif

namespace {
struct KernelArgs {
    // float Vr;  // TODO(priit) float vs double
    float prf;
    float lambda;
    float azimuth_bandwidth_fraction;
    bool apply_az_window;
    double range_spacing;
    double slant_range;

    float dc_poly[3];
    float Vr_poly[3];

    // bool vv_vh_half_pixel_shift;
};

template <int N>
__device__ inline float Polyval(float (&p)[N], float x) {
    float r = 0.0f;
    for (auto e : p) {
        r *= x;
        r += e;
    }
    return r;
}

__global__ void RangeCellMigrationCorrection(const cufftComplex* data_in, int src_x_size, int src_x_stride,
                                             int src_y_size, cufftComplex* data_out, int dst_x_stride,
                                             KernelArgs args) {
    const int dst_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int dst_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (dst_x >= src_x_size || dst_y >= src_y_size) {
        return;
    }

    const float fft_bin_step = args.prf / src_y_size;

    const float dc = Polyval(args.dc_poly, dst_x);

    // find FFT bin frequency
    float fn = 0.0f;
    if ((dst_y < src_y_size / 2)) {
        fn = dst_y * fft_bin_step;
    } else {
        fn = (dst_y - src_y_size) * fft_bin_step;
    }

    // example: doppler centroid -300Hz, prf 2000Hz, positive frequencies: 0-1000Hz =>
    //  frequency +800Hz baseband would be an alias to the actual frequency of -1200Hz
    const float half_prf = args.prf * 0.5f;
    if (dc < 0.0f) {
        if (fn > half_prf + dc) {
            fn -= args.prf;
        }
    } else if (dc > 0.0f) {
        if (fn < -half_prf + dc) {
            fn += args.prf;
        }
    }

    // slant range at range pixel
    const double R0 = args.slant_range + dst_x * args.range_spacing;

    const float Vr = Polyval(args.Vr_poly, dst_x);

    // Range walk in meters
    double dR = (args.lambda * args.lambda * fn * fn) / (8 * Vr * Vr);
    dR *= R0;
    // double dR = ((args.lambda * args.lambda) * R0 * fn * fn) / (8 * args.Vr * args.Vr);

    // range walk in pixels
    const float shift = dR / args.range_spacing;
    const int range_walk_pixels = (int)std::round(shift);
    const int dst_idx = dst_x + dst_y * dst_x_stride;

#if RCMC_INTERPOLATOR_SIZE == 0

    if (range_walk_pixels + dst_x < src_x_size) {
        int src_idx = dst_x + range_walk_pixels + dst_y * src_x_stride;
        data_out[dst_idx] = data_in[src_idx];
    } else {
        data_out[dst_idx] = {};
    }
#else
    const int N = N_INTERPOLATOR;
    float i_sum = 0.0f;
    float q_sum = 0.0f;
    float norm_sum = 0.0f;

    const int calc_x = dst_x + range_walk_pixels;
    const int first_idx = calc_x - N / 2 + dst_y * src_x_stride;

    for (int i = 0; i < N; i++) {
        if (calc_x - N / 2 < 0 || calc_x + N / 2 >= src_x_size) {
            continue;
        }

        const cufftComplex data = data_in[first_idx + i];

        const float sinc_offset = shift - range_walk_pixels + N / 2;

        const float mult = WINDOW[i] * sinc(sinc_offset - i);
        i_sum += data.x * mult;
        q_sum += data.y * mult;
        norm_sum += mult;
    }

    if (norm_sum == 0.0f) {
        data_out[dst_idx] = {};
    } else {
        data_out[dst_idx] = {i_sum / norm_sum, q_sum / norm_sum};
    }
#endif
}

__global__ void AzimuthReferenceMultiply(cufftComplex* src_data, int src_width, int src_height, int src_x_stride,
                                         KernelArgs args) {
    int dst_x = threadIdx.x + blockIdx.x * blockDim.x;
    int dst_y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((dst_x < src_width && dst_y < src_height)) {
        int src_idx = dst_x + src_x_stride * dst_y;

        const float fft_bin_step = args.prf / src_height;

        const float dc = Polyval(args.dc_poly, dst_x);

        const float dc_steps = std::roundf(dc / fft_bin_step);
        const float dc_offset = dc_steps * fft_bin_step;
        // find FFT bin frequency
        float bin_freq = 0.0f;
        if ((dst_y < src_height / 2)) {
            bin_freq = dst_y * fft_bin_step;

        } else {
            bin_freq = (dst_y - src_height) * fft_bin_step;
        }

        // TODO is this usage of the doppler centroid correct?
        // Multiple issues at play here:
        // throw away bin locations
        // matched filter center vs zero frequency - otherwise ghost in azimuth direction
        // the resulting azimuth time domain shift and it's undoing by a linear ramp shift
        // needs further investigation
        const float half_prf = args.prf * 0.5f;
        float abs_freq = bin_freq;
        if (dc < 0.0f) {
            if (bin_freq > half_prf + dc) {
                abs_freq -= args.prf;
            }
        } else if (dc > 0.0f) {
            if (abs_freq < -half_prf + dc) {
                abs_freq += args.prf;
            }
        }

        const float cutoff_hi = half_prf * args.azimuth_bandwidth_fraction + dc_offset;
        const float cutoff_lo = -half_prf * args.azimuth_bandwidth_fraction + dc_offset;

        if (abs_freq < cutoff_lo || abs_freq > cutoff_hi) {
            // frequency in the throwaway region
            // ex prf = 2000Hz, dc = -600Hz, fraction = 0.8 => cut low = +-1400Hz, cut high = +200Hz
            src_data[src_idx] = {};
            return;
        }

        // Adjust center frequency by doppler centroid, however keep the values in +- prf/2 for the matched filter
        float fn_center = bin_freq - dc_offset;
        if (fn_center > half_prf) {
            fn_center -= args.prf;
        } else if (fn_center < -half_prf) {
            fn_center += args.prf;
        }

        // slant range at range pixel
        const double R0 = args.slant_range + args.range_spacing * dst_x;

        const float Vr = Polyval(args.Vr_poly, dst_x);
        // Azimuth FM rate
        const float Ka = (2 * Vr * Vr) / (args.lambda * R0);

        // range Doppler domain matched filter at this frequency
        const float float_pi = M_PI;
        float mf_phase = (float_pi * fn_center * fn_center) / Ka;

        // exp((-1j* pi * fn * fn) / Ka)
        cufftComplex mf_val;
        {
            float sin_val;
            float cos_val;
            sincos(mf_phase, &sin_val, &cos_val);
            mf_val.x = cos_val;
            mf_val.y = -sin_val;
        }

        // the previous matched filter is not centered around zero due to doppler centroid
        // a linear phase ramp should put this back to 0Hz, aka zero doppler position
        // exp(-1j * 2 * pi * dt * fn)
        cufftComplex shift;
        {
            float shift_phase = 2 * float_pi * (dc_offset / Ka) * bin_freq;  // bin freq or dc offset frequency?
            float sin_val;
            float cos_val;
            sincos(shift_phase, &sin_val, &cos_val);
            shift.x = cos_val;
            shift.y = -sin_val;
        }

        mf_val = cuCmulf(mf_val, shift);

        /* The place to do azimuth windowing, however this needs further investigation
        if (args.apply_az_window) {
            constexpr float alpha = 0.54f;  // hamming
            constexpr float two_pi = 2 * M_PI;

            // TODO investigate if this correct way to do frequency domain weighting, as discussed in
            //  6.3.6 in Digital Processing of SAR Data
            float weight = alpha - (1.0f - alpha) * cos(two_pi * (abs_freq - dc) / args.prf);
            weight = 1.0f - weight;
            mf_val.x *= weight;
            mf_val.y *= weight;
        }
         */

        // application via f domain multiplication
        src_data[src_idx] = cuCmulf(src_data[src_idx], mf_val);
    }
}
}  // namespace

void RangeDopplerAlgorithm(const SARMetadata& metadata, DevicePaddedImage& src_img, DevicePaddedImage& out_img,
                           CudaWorkspace& d_workspace) {
#if INPLACE_AZIMUTH_FFT == 0
    {
        // 2D time domain -> range Doppler domain via Azimuth FFT
        // however, transpose -> range FFT -> transpose is faster than direct Azimuth FFT on all GPUs tested so far
        src_img.Transpose(d_workspace);
        auto azimuth_fft = PlanRangeFFT(src_img.XStride(), src_img.YSize(), false);
        CufftPlanCleanup fft_cleanup(azimuth_fft);
        alus::cuda::CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, src_img.Data(), src_img.Data(), CUFFT_FORWARD));
        // TODO(priit) not actually needed, can fixed by inverting indexing in RCMC and Azimuth Ref multiply
        src_img.Transpose(d_workspace);
    }
#else
    {
        auto azimuth_fft = PlanAzimuthFFT(src_img.XSize(), src_img.YStride(), src_img.XStride(), false);
        CufftPlanCleanup fft_cleanup(azimuth_fft);
        CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, src_img.Data(), src_img.Data(), CUFFT_FORWARD));
    }
#endif

    KernelArgs k_args = {};
    k_args.lambda = metadata.wavelength;
    k_args.range_spacing = metadata.range_spacing;
    // k_args.inv_range_spacing = 1 / metadata.range_spacing;
    k_args.slant_range = metadata.slant_range_first_sample;
    const auto& Vr_poly = metadata.results.Vr_poly;
    CHECK_BOOL(Vr_poly.size() == std::size(k_args.Vr_poly));
    std::copy(Vr_poly.begin(), Vr_poly.end(), &k_args.Vr_poly[0]);

    const auto& dc_poly = metadata.results.doppler_centroid_poly;
    CHECK_BOOL(dc_poly.size() == std::size(k_args.dc_poly));
    std::copy(dc_poly.begin(), dc_poly.end(), &k_args.dc_poly[0]);

    k_args.prf = metadata.pulse_repetition_frequency;
    // k_args.doppler_centroid = metadata.results.doppler_centroid;
    k_args.azimuth_bandwidth_fraction = metadata.azimuth_bandwidth_fraction;
    k_args.apply_az_window = metadata.azimuth_window;

    // double dc = CalcDopplerCentroid(metadata, metadata.img.range_size / 2);

    // LOGV << "KARGS prf = " << k_args.prf;
    // LOGV << "Doppler centroid steps = " << k_args.dc_steps;

    // give d_workspace memory to out_img
    out_img.InitExtPtr(src_img.XSize(), src_img.YSize(), src_img.XStride(), src_img.YStride(), d_workspace);

    out_img.ZeroMemory();

    dim3 block_size(16, 16);
    dim3 grid_size((src_img.XSize() + 15) / 16, (src_img.YStride() + 15) / 16);

    // RCMC step
    RangeCellMigrationCorrection<<<grid_size, block_size>>>(src_img.Data(), src_img.XSize(), src_img.XStride(),
                                                            src_img.YStride(), out_img.Data(), out_img.XStride(),
                                                            k_args);

    // Azimuth Compression step
    AzimuthReferenceMultiply<<<grid_size, block_size>>>(out_img.Data(), out_img.XSize(), out_img.YStride(),
                                                        out_img.XStride(), k_args);

    // src memory no longer needed and can now be used for workspace
    d_workspace = src_img.ReleaseMemory();

    // range Doppler domain back to 2D time domain
#if INPLACE_AZIMUTH_FFT == 0
    {
        // tranpose -> Range IFFT -> transpose instead of direct Azimuth IFFT
        out_img.Transpose(d_workspace);
        auto azimuth_fft = PlanRangeFFT(out_img.XStride(), out_img.YSize(), false);
        CufftPlanCleanup fft_cleanup(azimuth_fft);
        alus::cuda::CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, out_img.Data(), out_img.Data(), CUFFT_INVERSE));
        out_img.Transpose(d_workspace);
    }
#else
    {
        auto azimuth_fft = PlanAzimuthFFT(out_img.XSize(), out_img.YStride(), out_img.XStride(), false);
        CufftPlanCleanup fft_cleanup(azimuth_fft);
        CheckCufftSize(d_workspace.ByteSize(), azimuth_fft);
        CHECK_CUFFT_ERR(cufftSetWorkArea(azimuth_fft, d_workspace.Get()));
        CHECK_CUFFT_ERR(cufftExecC2C(azimuth_fft, out_img.Data(), out_img.Data(), CUFFT_INVERSE));
    }
#endif

    // scale FFT -> IFFT roundtrip
    out_img.MultiplyData(1.0f / out_img.YStride());

    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

namespace alus::sar::focus {

RcmcParameters GetRcmcParameters() {
    RcmcParameters params{};
    params.window_size = N_INTERPOLATOR;
    return params;
}

}
