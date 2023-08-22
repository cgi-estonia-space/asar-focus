#include <filesystem>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
//#include <boost/algorithm/algorithm.hpp>

#include "processing_velocity_estimation.h"
#include <boost/algorithm/string.hpp>
#include <gdal/gdal_priv.h>


#include "orbit_state_vector.h"
#include "geo_tools.h"
#include "cufft_plan.h"
#include "checks.h"
#include <complex>

#include "fractional_doppler_centroid.cuh"

#include "device_padded_image.cuh"
#include "envisat_file_format.h"
#include "asar_lvl1_file.h"

#include "sar_chirp.h"

#include "range_compression.cuh"

#include "img_output.h"

#include "range_doppler_algorithm.cuh"

#include "iq_correction.cuh"

#include "envisat_ins_file.h"




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



    if (argc != 3) {

        //to arg parse

        printf("error: arg 1 => input file, arg 2 => aux path");
        return 1;
    }


    std::string in_path = argv[1];
    const char* aux_path = argv[2];
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
    fclose(fp);

    GDALAllRegister();


    SARMetadata metadata = {};
    ASARMetadata asar_meta = {};



    std::vector<std::complex<float>> h_data;
    ParseIMFile(data, aux_path, metadata, asar_meta, h_data);

    double Vr = EstimateProcessingVelocity(metadata);

    metadata.results.Vr = 7097.110219566418;

    printf("Vr = %f\n", Vr);


    std::string wif_name_base = asar_meta.product_name;
    wif_name_base.resize(wif_name_base.size() - 3);


    int az_size = metadata.img.azimuth_size;
    int rg_size = metadata.img.range_size;
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






    auto chirp = GenerateChirpData(metadata.chirp, rg_padded);

    std::vector<float> chirp_freq;

    if(wif) {
        FILE *fpe = fopen("/tmp/chirp.cf32", "w");
        fwrite(chirp.data(), 8, chirp.size(), fpe);
        fclose(fpe);
    }


    DevicePaddedImage img;
    img.InitPadded(rg_size, az_size, rg_padded, az_padded);
    img.ZeroMemory();

    for(int y = 0; y < az_size; y++)
    {
        auto* dest = img.Data() + y * rg_padded;
        auto* src = h_data.data() + y * rg_size;
        cudaMemcpy(dest, src, rg_size * 8, cudaMemcpyHostToDevice);
    }

    // TODO
    //iq_correct_cuda(raw_adc, img);
    if (wif) {
        std::string path = "/tmp/";
        path += wif_name_base + "_raw.tif";
        WriteIntensityPaddedImg(img, path.c_str());
    }


    double dc = 0.0;
    CalculateDopplerCentroid(img, metadata.pulse_repetition_frequency, dc);

    metadata.results.doppler_centroid = -dc;


    printf("Image GPU byte size = %f GB\n", img.TotalByteSize() / 1e9);
    printf("Estimated GPU memory usage ~%f GB\n", (img.TotalByteSize() * 2) / 1e9);

    CudaWorkspace d_workspace(img.TotalByteSize());



    RangeCompression(img, chirp, metadata.chirp.n_samples, d_workspace);


    metadata.img.range_size = img.XSize();

    if (wif) {
        std::string path = "/tmp/";
        path += wif_name_base + "_rc.tif";
        WriteIntensityPaddedImg(img, path.c_str());
    }

    DevicePaddedImage out;


    RangeDopplerAlgorithm(metadata, img, out, std::move(d_workspace));


    out.ZeroFillPaddings();


    cudaDeviceSynchronize();

    if (wif) {
        std::string path = "/tmp/";
        path += wif_name_base + "_slc.tif";
        WriteIntensityPaddedImg(out, path.c_str());
    }


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
            //printf("y = %d\n", y);
            for (int x = 0; x < out.XSize(); x++) {
                auto pix = res[x + y * range_stride];
                int tambov = 1000;
                IQ16 iq16;
                iq16.i = std::clamp<float>(pix.x * tambov, INT16_MIN, INT16_MAX);
                iq16.q = std::clamp<float>(pix.y * tambov, INT16_MIN, INT16_MAX);

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

    WriteLvl1(metadata, asar_meta, mds);

    return 0;

}






