
#pragma once

#include "sar_metadata.h"

inline std::vector<std::complex<float>> GenerateChirpData(const ChirpInfo& chirp, size_t padding_size) {
    double Kr = chirp.Kr;

    const int64_t n_samples = chirp.n_samples;
    const double dt = 1.0 / chirp.range_sampling_rate;

    std::vector<std::complex<float>> chirp_data(n_samples);

    for (int64_t i = 0; i < n_samples; i++) {
        const double t = (i - n_samples / 2) * dt;
        const double phase = M_PI * Kr * t * t;
        std::complex<float> iq(cos(phase), sin(phase));

        iq /= sqrt(n_samples);

        chirp_data[i] = iq;
    }

    // ApplyWindowPad(chirp_data, padding_size);
    chirp_data.resize(padding_size);

    return chirp_data;
}