
#include "iq_correction.h"

#include <stdio.h>


void
iq_correct(const std::vector<IQ8> in, std::vector<std::complex<float>> &out, int rg_size, int rg_stride, int az_size,
           int az_stride) {

    double I = 0.0;
    double Q = 0.0;

    out.clear();
    out.resize(rg_stride * az_stride);


    printf("eq = %d\n", rg_size * az_size == in.size());
    size_t total_samples = in.size();
    size_t padded_size = rg_stride * az_stride;

    for (const auto &pixel: in) {
        I += pixel.i;
        Q += pixel.q;
    }


    float dc_i = I / total_samples;
    float dc_q = Q / total_samples;


    printf("DC I/Q = %f %f\n", dc_i, dc_q);
    printf("offset = %f %f\n", 15.5 - dc_i, 15.5 - dc_q);


    std::vector<std::complex<float>> dc_corrected(padded_size);

    double sq_sum_i = 0.0;
    double sq_sum_q = 0.0;

    int cnt = 0;
    for (int y = 0; y < az_size; y++) {
        int x = 0;
        for (; x < rg_size; x++) {
            auto iq8 = in[x + y * rg_size];
            float corr_i = iq8.i - dc_i;
            float corr_q = iq8.q - dc_q;
            if(x == 4000 && y == 25000)
            {
                printf("4k,25k pc iq8 = %d %d\n", iq8.i, iq8.q);
            }
            dc_corrected[x + y * rg_stride] = {corr_i, corr_q};
            sq_sum_i += corr_i * corr_i;
            sq_sum_q += corr_q * corr_q;
            cnt++;
        }
    }

    printf("gain cnt = %d\n", cnt);
    //out = std::move(dc_corrected); return;

    // gain correction
    printf("gain sums = %f %f\n", sq_sum_i, sq_sum_q);
    float gain = sqrt(sq_sum_i / sq_sum_q);

    printf("GAIN x86 I / Q = %f\n", gain);


    for (int y = 0; y < az_size; y++) {
        for (int x = 0; x < rg_size; x++) {
            auto &pix = dc_corrected[x + y * rg_stride];
            pix.real(pix.real() / gain);
        }
    }


#if 1
    double iq_mul = 0.0;
    double i_sq = 0.0;
    double q_sq = 0.0;

    for (int y = 0; y < az_size; y++) {
        for (int x = 0; x < rg_size; x++) {
            const auto pix = dc_corrected[x + y * rg_stride];
            float i = pix.real();
            float q = pix.imag();
            iq_mul += i * q;
            i_sq += i * i;
            q_sq += q * q;
        }
    }

    double c0 = iq_mul / i_sq;
    double c1 = iq_mul / q_sq;

    printf("%f %f\n", c0, c1);

    double quad_misconf = asin(c0);

    printf("quadruture depart: %f\n", quad_misconf);

    printf("phase offset = %f\n", (quad_misconf * 360.0) / (M_PI * 2));

    float sin_corr = c0;
    float cos_corr = sqrt(1 - c0 * c0);

    for (int y = 0; y < az_size; y++) {
        for (int x = 0; x < rg_size; x++) {
            auto &pix = dc_corrected[x + y * rg_stride];
            float i = pix.real();
            float q = pix.imag();
            float corr_q = (q - i * sin_corr) / cos_corr;
            pix.imag(corr_q);
        }
    }
#endif

    out = std::move(dc_corrected);
    return;

}