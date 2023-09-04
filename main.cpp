#include <stdio.h>
#include <complex>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gdal/gdal_priv.h>
#include <boost/algorithm/string.hpp>

#include "cuda_util/cufft_plan.h"
#include "cuda_util/device_padded_image.cuh"
#include "envisat_format/envisat_aux_file.h"
#include "envisat_format/envisat_lvl1_writer.h"
#include "envisat_format/envisat_mph_sph_parser.h"
#include "sar/fractional_doppler_centroid.cuh"
#include "sar/iq_correction.cuh"
#include "sar/orbit_state_vector.h"
#include "sar/processing_velocity_estimation.h"
#include "sar/range_compression.cuh"
#include "sar/range_doppler_algorithm.cuh"
#include "sar/sar_chirp.h"
#include "util/checks.h"
#include "util/geo_tools.h"
#include "util/img_output.h"
#include "util/math_utils.h"
#include "util/plot.h"

struct IQ16 {
    int16_t i;
    int16_t q;
};

bool wif = false;

auto time_start() { return std::chrono::steady_clock::now(); }

void time_stop(std::chrono::steady_clock::time_point beg, const char* msg) {
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
    std::cout << msg << " time = " << diff << " ms\n";
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        // to arg parse

        printf("Not a correct argument count, example:\n"
               "%s [input file] [aux path] [DORIS orbit file/folder]", argv[0]);
        return 1;
    }

    auto file_time_start = time_start();
    std::string in_path = argv[1];
    const char* aux_path = argv[2];
    const auto orbit_path = argv[3];
    FILE* fp = fopen(in_path.c_str(), "r");
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

    auto orbit_source = alus::dorisorbit::Parsable::TryCreateFrom(orbit_path);

    std::vector<std::complex<float>> h_data;
    ParseIMFile(data, aux_path, metadata, asar_meta, h_data, orbit_source);

    {
        const auto& orbit_l1_metadata = orbit_source.GetL1ProductMetadata();
        asar_meta.orbit_metadata.vector_source = orbit_l1_metadata.vector_source;
        asar_meta.orbit_metadata.x_position = orbit_l1_metadata.x_position;
        asar_meta.orbit_metadata.y_position = orbit_l1_metadata.y_position;
        asar_meta.orbit_metadata.z_position = orbit_l1_metadata.z_position;
        asar_meta.orbit_metadata.x_velocity = orbit_l1_metadata.x_velocity;
        asar_meta.orbit_metadata.y_velocity = orbit_l1_metadata.y_velocity;
        asar_meta.orbit_metadata.z_velocity = orbit_l1_metadata.z_velocity;
        asar_meta.orbit_metadata.state_vector_time = orbit_l1_metadata.state_vector_time;
        asar_meta.orbit_metadata.phase = orbit_l1_metadata.phase;
        asar_meta.orbit_metadata.cycle = orbit_l1_metadata.cycle;
        asar_meta.orbit_metadata.abs_orbit = orbit_l1_metadata.abs_orbit;
        asar_meta.orbit_metadata.orbit_name = orbit_l1_metadata.orbit_name;
    }

    time_stop(file_time_start, "LVL0 file read + parse");

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
           az_plan_sz->second[0], az_plan_sz->second[1], az_plan_sz->second[2], az_plan_sz->second[3]);
    int az_padded = az_plan_sz->first;

    printf("FFT paddings:\n");
    printf("rg  = %d -> %d\n", rg_size, rg_padded);
    printf("az = %d -> %d\n", az_size, az_padded);


    auto gpu_transfer_start = time_start();
    DevicePaddedImage img;
    img.InitPadded(rg_size, az_size, rg_padded, az_padded);
    img.ZeroMemory();

    for (int y = 0; y < az_size; y++) {
        auto* dest = img.Data() + y * rg_padded;
        auto* src = h_data.data() + y * rg_size;
        CHECK_CUDA_ERR(cudaMemcpy(dest, src, rg_size * 8, cudaMemcpyHostToDevice));
    }

    time_stop(gpu_transfer_start, "GPU image formation");

    auto correction_start = time_start();
    RawDataCorrection(img, {metadata.total_raw_samples}, metadata.results);
    img.ZeroNaNs();
    time_stop(correction_start, "I/Q correction");


    auto vr_start = time_start();
    metadata.results.Vr_poly = EstimateProcessingVelocity(metadata);
    time_stop(vr_start, "Vr estimation");


    // if(wif)
    {
        PlotArgs args = {};
        args.out_path = "/tmp/" + wif_name_base + "_Vr.html";
        args.x_axis_title = "range index";
        args.y_axis_title = "Vr(m/s)";
        args.data.resize(1);
        auto& line = args.data[0];
        line.line_name = "Vr";
        PolyvalRange(metadata.results.Vr_poly, 0, metadata.img.range_size, line.x, line.y);
        Plot(args);
    }

    auto chirp = GenerateChirpData(metadata.chirp, rg_padded);

    std::vector<float> chirp_freq;

    // if (wif)
    {
        PlotArgs args = {};
        args.out_path = "/tmp/" + wif_name_base + "_chirp.html";
        args.x_axis_title = "nth sample";
        args.y_axis_title = "Amplitude";
        args.data.resize(2);
        auto& i = args.data[0];
        auto& q = args.data[1];
        std::vector<double> n_samp;
        int cnt = 0;
        for (auto iq : chirp) {
            i.y.push_back(iq.real());
            i.x.push_back(cnt);
            q.y.push_back(iq.imag());
            q.x.push_back(cnt);
            cnt++;
        }

        i.line_name = "I";
        q.line_name = "Q";
        Plot(args);
    }

    if (wif) {
        std::string path = "/tmp/";
        path += wif_name_base + "_raw.tif";
        WriteIntensityPaddedImg(img, path.c_str());
    }

    auto dc_start = time_start();
    metadata.results.doppler_centroid_poly = CalculateDopplerCentroid(img, metadata.pulse_repetition_frequency);
    time_stop(dc_start, "fractional DC estimation");
    // if(wif)
    {
        PlotArgs args = {};
        args.out_path = "/tmp/" + wif_name_base + "_dc.html";
        args.x_axis_title = "range index";
        args.y_axis_title = "Hz";
        args.data.resize(1);
        auto& line = args.data[0];
        line.line_name = "Doppler centroid";
        PolyvalRange(metadata.results.doppler_centroid_poly, 0, metadata.img.range_size, line.x, line.y);
        Plot(args);
    }

    // return 0;

    printf("Image GPU byte size = %f GB\n", img.TotalByteSize() / 1e9);
    printf("Estimated GPU memory usage ~%f GB\n", (img.TotalByteSize() * 2) / 1e9);

    CudaWorkspace d_workspace(img.TotalByteSize());

    auto rc_start = time_start();
    RangeCompression(img, chirp, metadata.chirp.n_samples, d_workspace);
    time_stop(rc_start, "Range compression");

    metadata.img.range_size = img.XSize();

    auto src_start = time_start();
    SecondaryRangeCompression(img, metadata, d_workspace);
    time_stop(src_start, "Secondary Range compression");

    if (wif || true) {
        std::string path = "/tmp/";
        path += wif_name_base + "_src.tif";
        WriteIntensityPaddedImg(img, path.c_str());
    }



    if (wif) {
        std::string path = "/tmp/";
        path += wif_name_base + "_rc.tif";
        WriteIntensityPaddedImg(img, path.c_str());
    }

    auto az_comp_start = time_start();
    DevicePaddedImage out;
    RangeDopplerAlgorithm(metadata, img, out, std::move(d_workspace));


    out.ZeroFillPaddings();

    time_stop(az_comp_start, "Azimuth compression");
    cudaDeviceSynchronize();

    if (wif || true) {
        std::string path = "/tmp/";
        path += wif_name_base + "_slc.tif";
        WriteIntensityPaddedImg(out, path.c_str());
    }
    return 0;

    printf("done!\n");

    auto cpu_transfer_start = time_start();
    cufftComplex* res = new cufftComplex[out.XStride() * out.YStride()];
    out.CopyToHostPaddedSize(res);

    time_stop(cpu_transfer_start, "Image GPU->CPU");
    // metadata.


    auto mds_formation = time_start();
    MDS mds;
    mds.n_records = out.YSize();
    mds.record_size = out.XSize() * 4 + 12 + 1 + 4;
    mds.buf = new char[mds.n_records * mds.record_size];

    {
        int range_size = out.XSize();
        int range_stride = out.XStride();
        std::vector<IQ16> row(range_size);
        for (int y = 0; y < out.YSize(); y++) {
            // printf("y = %d\n", y);
            for (int x = 0; x < out.XSize(); x++) {
                auto pix = res[x + y * range_stride];
                float tambov = 120000/100;
                //printf("scaling tambov = %f\n", tambov);
                IQ16 iq16;
                iq16.i = std::clamp<float>(pix.x * tambov, INT16_MIN, INT16_MAX);
                iq16.q = std::clamp<float>(pix.y * tambov, INT16_MIN, INT16_MAX);

                iq16.i = bswap(iq16.i);
                iq16.q = bswap(iq16.q);
                row[x] = iq16;
            }

            memset(&mds.buf[y * mds.record_size], 0, 17); //TODO each row mjd
            memcpy(&mds.buf[y * mds.record_size + 17], row.data(), row.size() * 4);
        }
    }
    time_stop(mds_formation, "MDS construction");

    auto file_write = time_start();
    WriteLvl1(metadata, asar_meta, mds);
    time_stop(file_write, "LVL1 file write");


    return 0;
}
