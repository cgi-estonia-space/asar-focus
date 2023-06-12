#include <filesystem>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
//#include <boost/algorithm/algorithm.hpp>
#include <boost/algorithm/string.hpp>
#include <gdal/gdal_priv.h>


#include "cufft_plan.h"
#include "checks.h"
#include <complex>

#include "fractional_doppler_centroid.cuh"

#include "device_padded_image.cuh"
#include "envisat_file_format.h"
#include "ers_im_lvl1_file.h"

#include "sar_chirp.h"

#include "range_compression.cuh"
#include "iq_correction.h"
#include "img_output.h"

#include "range_doppler_algorithm.cuh"

#include "iq_correction.cuh"




struct IQ16
{
    int16_t i;
    int16_t q;
};


bool wif = false;

void LoadMetadata(const std::string &path, SARMetadata &metadata);


auto time_start() { return std::chrono::steady_clock::now(); }

void time_stop(std::chrono::steady_clock::time_point beg, const char *msg) {
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
    std::cout << msg << " time = " << diff << " ms\n";
}

void WriteOutput(SARMetadata& meta)
{

}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        return 1;
    }
    std::string in_path = argv[1];
    FILE *fp = fopen(in_path.c_str(), "r");
    if (!fp) {
        printf("failed to open ... %s\n", in_path.c_str());
        return 1;
    }
    size_t file_sz = std::filesystem::file_size(in_path.c_str());


    std::vector<char> data(file_sz);
    auto act_sz = fread(data.data(), 1, file_sz, fp);
    if (act_sz != file_sz) {
        printf("Failed to read %zu bytes.. expected = %zu!\n", file_sz, act_sz);
        return 1;
    }
    //fclose(fp);

    GDALAllRegister();

    ProductHeader mph = {};
    mph.set(0, "MPH", data.data(), MPH_SIZE);

    mph.print();

    ProductHeader specific_ph = {};
    specific_ph.set(MPH_SIZE, "sph", data.data() + MPH_SIZE, SPH_SIZE);

    specific_ph.print();


    auto dsds = extract_dsds(specific_ph);


    auto source_packets = dsds[0];

    int az_size = source_packets.num_dsr;
    const int rg_size = 11232 / 2;

    printf("Rg, Az size = %d %d\n", rg_size, az_size);


    CHECK_BOOL(source_packets.dsr_size == 11498);
    CHECK_BOOL(source_packets.ds_offset == MPH_SIZE + SPH_SIZE);
    CHECK_BOOL(source_packets.ds_offset + source_packets.ds_size == file_sz);

    constexpr uint32_t signal_offset = 32 + 4 + 10 + 220;

    static_assert(signal_offset + rg_size * 2 == 11498);


    std::vector<IQ8> raw_adc(az_size * rg_size);

    char *first_mdsr = data.data() + source_packets.ds_offset;

    for (size_t i = 0; i < az_size; i++) {
        // 32
        char *dsr = first_mdsr + i * source_packets.dsr_size + signal_offset;
        //size_t iq8_idx = i * rg_size;
        IQ8 *dest = &raw_adc[i * rg_size];
        memcpy(dest, dsr, source_packets.dsr_size);
    }

    auto fft_sizes = GetOptimalFFTSizes();

    auto rg_plan_sz = fft_sizes.upper_bound(rg_size);
    printf("range FFT padding sz = %d (2 ^ %d * 3 ^ %d * 5 ^ %d * 7 ^ %d)\n", rg_plan_sz->first, rg_plan_sz->second[0],
           rg_plan_sz->second[1], rg_plan_sz->second[2], rg_plan_sz->second[3]);
    int rg_padded = rg_plan_sz->first;

    auto az_plan_sz = fft_sizes.upper_bound(az_size);

    printf("azimuth FFT padding sz = %d (2 ^ %d * 3 ^ %d * 5 ^ %d * 7 ^ %d)\n", az_plan_sz->first,
           az_plan_sz->second[0],
           az_plan_sz->second[1], az_plan_sz->second[2], az_plan_sz->second[3]);
    int az_padded = az_plan_sz->first;


    printf("FFT paddings:\n");
    printf("rg  = %d -> %d\n", rg_size, rg_padded);
    printf("az = %d -> %d\n", az_size, az_padded);


    SARMetadata metadata = {};

    LoadMetadata(in_path, metadata);

    metadata.carrier_frequency = 5.3e9;
    double frequency = 5.3e9;
    ChirpInfo ci = {};
    ci.range_sampling_rate = 1.8962468e7;
    ci.coefficient[1] = 2.08894001e11;
    ci.coefficient[1] *= 2;

    double echo_time = 2.9616398e-4;

    printf("echo time n samp = %f\n", echo_time / (1 / ci.range_sampling_rate));
    double pulse_time = 3.712597e-5;


    double n_samp = pulse_time / (1 / ci.range_sampling_rate);

    ci.n_samples = n_samp;


    std::cout << "n pulse =  " << n_samp << "\n";


    printf("%15.15f\n", n_samp);

    std::cout << "BW = " << ci.coefficient[1] * pulse_time << "\n";
    std::cout << "K = " << ci.coefficient[1] << "\n";

    metadata.chirp = ci;


    auto chirp = GenerateChirpData(ci, rg_padded);

    std::vector<float> chirp_freq;

    FILE *fpe = fopen("/tmp/ers_chirp.cf32", "w");
    fwrite(chirp.data(), 8, chirp.size(), fpe);
    fclose(fpe);


    DevicePaddedImage img;
    img.InitPadded(rg_size, az_size, rg_padded, az_padded);

    iq_correct_cuda(raw_adc, img);


    double dc = 0.0;
    CalculateDopplerCentroid(img, metadata.pulse_repetition_frequency, dc);

    metadata.results.doppler_centroid = 0;
    if (wif) {
        WriteIntensityPaddedImg(img, "/tmp/ers_raw.tif", false);
    }


    printf("Image GPU byte size = %f GB\n", img.TotalByteSize() / 1e9);
    printf("Estimated GPU memory usage ~%f GB\n", (img.TotalByteSize() * 2) / 1e9);

    CudaWorkspace d_workspace(img.TotalByteSize());


    RangeCompression(img, chirp, n_samp, d_workspace);

    if (wif) {
        WriteIntensityPaddedImg(img, "/tmp/ers_rc.tif", false);
    }


    DevicePaddedImage out;


    RangeDopplerAlgorithm(metadata, img, out, std::move(d_workspace));


    out.ZeroFillPaddings();


    cudaDeviceSynchronize();

    std::string path =
            "/tmp/ers_azcomp_intensity.tif";

    //WriteIntensityPaddedImg(out, path.c_str(), true);

    printf("done!\n");



    cufftComplex* res = new cufftComplex[out.XStride()*out.YStride()];
    out.CopyToHostPaddedSize(res);

    //metadata.

    MDS mds;

    mds.n_records = out.YSize();
    mds.record_size = out.XSize() * 4 + 12 + 1 + 4;

    mds.buf = new char[mds.n_records * mds.record_size];

    {
        int range_size = out.XSize();
        int range_stride = out.XStride();
        std::vector<IQ16> row(range_size);
        for (int y = 0; y < out.YSize(); y++) {
            printf("y = %d\n", y);
            for (int x = 0; x < out.XSize(); x++) {
                auto pix = res[x + y * range_stride];
                IQ16 iq16;
                iq16.i = std::clamp<float>(pix.x, INT16_MIN, INT16_MAX);
                iq16.q = std::clamp<float>(pix.y, INT16_MIN, INT16_MAX);
                iq16.i = bswap(iq16.i);
                iq16.q = bswap(iq16.q);
                row[x] = iq16;
            }

            memset(&mds.buf[y*mds.record_size], 0, 17);
            memcpy(&mds.buf[y*mds.record_size + 17], row.data(), row.size()*4);
        }

    }

    cudaDeviceSynchronize();

    WriteOutput(metadata);

    WriteLvl1(metadata, mds);

    return 0;

}






void LoadMetadata(const std::string &path, SARMetadata &metadata) {
    constexpr double c = 299792458.0;

    metadata.carrier_frequency = 5.3e9;
    metadata.range_spacing = 7.90489;
    metadata.wavelength = 299792458 / 5.3e9;
    if (path.find("SAR_IM__0PWDSI1995") != std::string::npos) {
        metadata.pulse_repetition_frequency = 1694.9176;
        double t = 5486194.5e-9;
        metadata.slant_range_first_sample = (t * c) / 2.0;
        double vx = 642686395e-5;
        double vy = 85775774e-5;
        double vz = -387046083e-5;
        metadata.platform_velocity = sqrt(vx*vx + vy*vy + vz*vz);

    } else if (path.find("SAR_IM__0PWDSI20040101") != std::string::npos) {
        metadata.pulse_repetition_frequency = 1679.9023;
        double t = 5578164.5e-9;
        metadata.slant_range_first_sample = (t * c) / 2.0;
        double vx = 659302430e-5;
        double vy = 52978616e-5;
        double vz = -362970804e-5;
        metadata.platform_velocity = sqrt(vx*vx + vy*vy + vz*vz);

    }
    else
    {
        printf("unsupported file... forw now..\n");
        exit(1);
    }

    printf("prf = %f Hz\n", metadata.pulse_repetition_frequency);
    printf("Vs = %f m/s\n", metadata.platform_velocity);
    printf("R0 = %f m\n", metadata.slant_range_first_sample);
    printf("Center frequency = %f\n", metadata.carrier_frequency);
    printf("wavelength = %f\n", metadata.wavelength);
}