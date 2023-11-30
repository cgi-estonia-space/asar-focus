/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include <gdal/gdal_priv.h>
#include <boost/algorithm/string.hpp>

#include "VERSION"
#include "alus_log.h"
#include "args.h"
#include "asar_constants.h"
#include "cuda_device_init.h"
#include "cufft_plan.h"
#include "device_padded_image.cuh"
#include "envisat_aux_file.h"
#include "envisat_lvl1_writer.h"
#include "envisat_types.h"
#include "file_async.h"
#include "geo_tools.h"
#include "img_output.h"
#include "main_flow.h"
#include "math_utils.h"
#include "mem_alloc.h"
#include "plot.h"
#include "sar/fractional_doppler_centroid.cuh"
#include "sar/iq_correction.cuh"
#include "sar/processing_velocity_estimation.h"
#include "sar/range_compression.cuh"
#include "sar/range_doppler_algorithm.cuh"
#include "sar/sar_chirp.h"

namespace {

template <typename T>
void ExceptionMessagePrint(const T& e) {
    LOGE << "Caught an exception";
    LOGE << e.what();
    LOGE << "Exiting.";
}

std::string GetSoftwareVersion() { return "asar_focus/" + std::string(VERSION_STRING); }

auto TimeStart() { return std::chrono::steady_clock::now(); }

void TimeStop(std::chrono::steady_clock::time_point beg, const char* msg) {
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
    LOGD << msg << " time = " << diff << " ms";
}
}  // namespace

int main(int argc, char* argv[]) {
    std::string args_help{};
    try {
        const std::vector<char*> args_raw(argv, argv + argc);
        alus::asar::Args args(args_raw);
        args_help = args.GetHelp();
        if (args.IsHelpRequested()) {
            std::cout << GetSoftwareVersion() << std::endl;
            std::cout << args.GetHelp() << std::endl;
            exit(0);
        }

        alus::asar::log::Initialize();
        alus::asar::log::SetLevel(args.GetLogLevel());
        auto cuda_init = alus::cuda::CudaInit();

        LOGI << GetSoftwareVersion();
        auto file_time_start = TimeStart();
        const auto in_path{args.GetInputDsPath()};
        FILE* fp = fopen(in_path.data(), "r");
        if (!fp) {
            LOGE << "Failed to open input - " << in_path;
            return 1;
        }

        size_t file_sz = std::filesystem::file_size(in_path.data());

        std::vector<char> data(file_sz);
        auto act_sz = fread(data.data(), 1, file_sz, fp);
        if (act_sz != file_sz) {
            LOGE << "Failed to read " << file_sz << " bytes, expected " << act_sz << " bytes from input dataset";
            return 1;
        }
        fclose(fp);
        TimeStop(file_time_start, "Input DS read and fetch");

        GDALAllRegister();

        SARMetadata metadata = {};
        ASARMetadata asar_meta = {};
        file_time_start = TimeStart();
        auto orbit_source = alus::dorisorbit::Parsable::TryCreateFrom(args.GetOrbitPath());
        TimeStop(file_time_start, "Orbit file fetch");
        file_time_start = TimeStart();
        alus::asar::envformat::ParseLevel0Header(data, asar_meta);
        const auto product_type = alus::asar::mainflow::TryDetermineProductType(asar_meta.product_name);
        alus::asar::mainflow::CheckAndLimitSensingStartEnd(asar_meta.sensing_start, asar_meta.sensing_stop,
                                                           args.GetProcessSensingStart(), args.GetProcessSensingEnd());
        alus::asar::mainflow::TryFetchOrbit(orbit_source, asar_meta, metadata);
        InstrumentFile ins_file{};
        ConfigurationFile conf_file{};
        alus::asar::mainflow::FetchAuxFiles(ins_file, conf_file, asar_meta, product_type, args.GetAuxPath());
        const auto target_product_type =
            alus::asar::specification::TryDetermineTargetProductFrom(product_type, args.GetFocussedProductType());
        (void)target_product_type;
        while (!cuda_init.IsFinished())
            ;
        cuda_init.CheckErrors();
        cufftComplex* d_parsed_packets;
        alus::asar::envformat::ParseLevel0Packets(data, metadata, asar_meta, &d_parsed_packets, product_type, ins_file,
                                                  asar_meta.sensing_start, asar_meta.sensing_stop);

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
            asar_meta.orbit_metadata.delta_ut1 = orbit_l1_metadata.delta_ut1;
            asar_meta.orbit_dataset_name = orbit_l1_metadata.orbit_name;
        }

        TimeStop(file_time_start, "LVL0 file packet parse");

        std::string wif_name_base = asar_meta.product_name;
        wif_name_base.resize(wif_name_base.size() - 3);

        const int az_size = metadata.img.azimuth_size;
        const int rg_size = metadata.img.range_size;
        const auto fft_sizes = GetOptimalFFTSizes();

        const auto rg_plan_sz = fft_sizes.upper_bound(rg_size + metadata.chirp.n_samples);
        LOGD << "Range FFT padding sz = " << rg_plan_sz->first << "(2 ^ " << rg_plan_sz->second[0] << " * 3 ^ "
             << rg_plan_sz->second[1] << " * 5 ^ " << rg_plan_sz->second[2] << " * 7 ^ " << rg_plan_sz->second[3]
             << ")";
        const int rg_padded = rg_plan_sz->first;

        const auto az_plan_sz = fft_sizes.upper_bound(az_size);

        LOGD << "azimuth FFT padding sz = " << az_plan_sz->first << " (2 ^ " << az_plan_sz->second[0] << " * 3 ^ "
             << az_plan_sz->second[1] << " * 5 ^ " << az_plan_sz->second[2] << " * 7 ^ " << az_plan_sz->second[3]
             << ")";
        int az_padded = az_plan_sz->first;

        LOGD << "FFT paddings: rg  = " << rg_size << "->" << rg_padded << " az = " << az_size << "->" << az_padded;

        auto gpu_transfer_start = TimeStart();
        DevicePaddedImage img;
        img.InitPadded(rg_size, az_size, rg_padded, az_padded);
        img.ZeroMemory();

        CHECK_CUDA_ERR(
            cudaMemcpy2D(img.Data(), rg_padded * 8, d_parsed_packets, rg_size * 8, rg_size * 8, az_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERR(cudaFree(d_parsed_packets));

        TimeStop(gpu_transfer_start, "GPU image formation");

        auto correction_start = TimeStart();
        RawDataCorrection(img, {metadata.total_raw_samples}, metadata.results);
        img.ZeroNaNs();
        TimeStop(correction_start, "I/Q correction");

        auto vr_start = TimeStart();
        metadata.results.Vr_poly = EstimateProcessingVelocity(metadata);
        TimeStop(vr_start, "Vr estimation");

        if (args.StorePlots()) {
            PlotArgs plot_args = {};
            plot_args.out_path = std::string(args.GetOutputPath()) + "/" + wif_name_base + "_Vr.html";
            plot_args.x_axis_title = "range index";
            plot_args.y_axis_title = "Vr(m/s)";
            plot_args.data.resize(1);
            auto& line = plot_args.data[0];
            line.line_name = "Vr";
            PolyvalRange(metadata.results.Vr_poly, 0, metadata.img.range_size, line.x, line.y);
            Plot(plot_args);
        }

        auto chirp = GenerateChirpData(metadata.chirp, rg_padded);

        std::vector<float> chirp_freq;

        if (args.StorePlots()) {
            PlotArgs plot_args = {};
            plot_args.out_path = std::string(args.GetOutputPath()) + "/" + wif_name_base + "_chirp.html";
            plot_args.x_axis_title = "nth sample";
            plot_args.y_axis_title = "Amplitude";
            plot_args.data.resize(2);
            auto& i = plot_args.data[0];
            auto& q = plot_args.data[1];
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
            Plot(plot_args);
        }

        if (args.StoreIntensity()) {
            std::string path = std::string(args.GetOutputPath()) + "/";
            path += wif_name_base + "_raw.tif";
            WriteIntensityPaddedImg(img, path.c_str());
        }

        auto dc_start = TimeStart();
        metadata.results.doppler_centroid_poly = CalculateDopplerCentroid(img, metadata.pulse_repetition_frequency);
        TimeStop(dc_start, "fractional DC estimation");
        if (args.StorePlots()) {
            PlotArgs plot_args = {};
            plot_args.out_path = std::string(args.GetOutputPath()) + "/" + wif_name_base + "_dc.html";
            plot_args.x_axis_title = "range index";
            plot_args.y_axis_title = "Hz";
            plot_args.data.resize(1);
            auto& line = plot_args.data[0];
            line.line_name = "Doppler centroid";
            PolyvalRange(metadata.results.doppler_centroid_poly, 0, metadata.img.range_size, line.x, line.y);
            Plot(plot_args);
        }

        LOGD << "Image GPU byte size = " << (img.TotalByteSize() / 1e9) << "GB";
        LOGD << "Estimated GPU memory usage ~" << ((img.TotalByteSize() * 2) / 1e9) << "GB";

        CudaWorkspace d_workspace(img.TotalByteSize());

        auto rc_start = TimeStart();
        RangeCompression(img, chirp, metadata.chirp.n_samples, d_workspace);
        TimeStop(rc_start, "Range compression");

        metadata.img.range_size = img.XSize();

        if (args.StoreIntensity()) {
            std::string path = std::string(args.GetOutputPath()) + "/";
            path += wif_name_base + "_rc.tif";
            WriteIntensityPaddedImg(img, path.c_str());
        }

        constexpr size_t record_header_bytes = 12 + 1 + 4;
        const auto mds_record_size = img.XSize() * sizeof(IQ16) + record_header_bytes;
        const auto mds_record_count = img.YSize();
        MDS mds;
        mds.n_records = mds_record_count;
        mds.record_size = mds_record_size;
        auto mds_buffer_init = std::async(std::launch::async, [&mds] {
            mds.buf = static_cast<char*>(alus::util::Memalloc(mds.n_records * mds.record_size));
        });

        std::string lvl1_out_name = asar_meta.product_name;
        lvl1_out_name.at(6) = 'S';
        lvl1_out_name.at(8) = '1';
        const std::string lvl1_out_full_path =
            std::string(args.GetOutputPath()) + std::filesystem::path::preferred_separator + lvl1_out_name;
        auto lvl1_file_handle = alus::util::FileAsync(lvl1_out_full_path);

        auto az_comp_start = TimeStart();
        DevicePaddedImage out;
        RangeDopplerAlgorithm(metadata, img, out, d_workspace);

        out.ZeroFillPaddings();

        EnvisatIMS ims{};
        ConstructIMS(ims, metadata, asar_meta, mds, GetSoftwareVersion());
        if (!lvl1_file_handle.CanWrite(std::chrono::seconds(10))) {
            throw std::runtime_error("Could not create L1 file at " + lvl1_out_full_path);
        }
        lvl1_file_handle.Write(&ims, sizeof(ims));

        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        TimeStop(az_comp_start, "Azimuth compression");

        if (args.StoreIntensity()) {
            std::string path = std::string(args.GetOutputPath()) + "/";
            path += wif_name_base + "_slc.tif";
            WriteIntensityPaddedImg(out, path.c_str());
        }

        auto result_correction_start = TimeStart();
        if (d_workspace.ByteSize() < static_cast<size_t>(mds.record_size * mds.n_records)) {
            throw std::logic_error(
                "Implementation error occurred - CUDA workspace buffer shall be made larger or equal to what is "
                "required for MDS buffer.");
        }
        constexpr float tambov{120000 / 100};
        auto device_mds_buf = d_workspace.GetAs<char>();
        alus::asar::mainflow::FormatResults(out, device_mds_buf, record_header_bytes, tambov);
        TimeStop(result_correction_start, "Image results correction");

        auto mds_formation = TimeStart();

        if (!mds_buffer_init.valid()) {
            throw std::runtime_error("MDS output buffer initialization went into invalid state");
        }
        const auto mds_buffer_init_result = mds_buffer_init.wait_for(std::chrono::seconds(10));
        if (mds_buffer_init_result != std::future_status::ready) {
            throw std::runtime_error(
                "Initializing MDS buffer failed. Please check the platform. The timeout 10 seconds was reached.");
        }
        mds_buffer_init.get();

        CHECK_CUDA_ERR(cudaMemcpy(mds.buf, device_mds_buf, mds.n_records * mds.record_size, cudaMemcpyDeviceToHost));
        TimeStop(mds_formation, "MDS Host buffer transfer");

        auto file_write = TimeStart();
        if (!lvl1_file_handle.CanWrite(std::chrono::seconds(10))) {
            throw std::runtime_error("Writing headers and DSDs of Level 1 product at " + lvl1_out_full_path +
                                     " did not finish in 10 seconds. Check platform for the performance issues.");
        }
        lvl1_file_handle.WriteSync(mds.buf, mds.n_records * mds.record_size);
        TimeStop(file_write, "LVL1 file write");
    } catch (const boost::program_options::error& e) {
        ExceptionMessagePrint(e);
        std::cout << args_help << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        ExceptionMessagePrint(e);
        exit(2);
    } catch (...) {
        LOGE << "Unknown exception occured";
        LOGE << "Exiting";
        exit(2);
    }

    return 0;
}
