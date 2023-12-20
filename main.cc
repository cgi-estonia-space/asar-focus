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

        std::vector<char> l0_ds(file_sz);
        auto act_sz = fread(l0_ds.data(), 1, file_sz, fp);
        if (act_sz != file_sz) {
            LOGE << "Failed to read " << file_sz << " bytes, expected " << act_sz << " bytes from input dataset";
            return 1;
        }
        fclose(fp);
        TimeStop(file_time_start, "Input DS read and fetch");

        GDALAllRegister();

        SARMetadata sar_meta = {};
        ASARMetadata asar_meta = {};
        file_time_start = TimeStart();
        auto orbit_source = alus::dorisorbit::Parsable::TryCreateFrom(args.GetOrbitPath());
        TimeStop(file_time_start, "Orbit file fetch");
        file_time_start = TimeStart();
        alus::asar::envformat::ParseLevel0Header(l0_ds, asar_meta);
        const auto product_type = alus::asar::mainflow::TryDetermineProductType(asar_meta.product_name);
        alus::asar::mainflow::CheckAndLimitSensingStartEnd(asar_meta.sensing_start, asar_meta.sensing_stop,
                                                           args.GetProcessSensingStart(), args.GetProcessSensingEnd());
        alus::asar::mainflow::TryFetchOrbit(orbit_source, asar_meta, sar_meta);
        InstrumentFile ins_file{};
        ConfigurationFile conf_file{};
        alus::asar::envformat::aux::ExternalCalibration xca;
        alus::asar::mainflow::FetchAuxFiles(ins_file, conf_file, xca, asar_meta, product_type, args.GetAuxPath());
        const auto target_product_type =
            alus::asar::specification::TryDetermineTargetProductFrom(product_type, args.GetFocussedProductType());
        (void)target_product_type;
        // Get packets' sensing start/end ... from which can be decided offset in binary and windowing offset?
        const auto fetch_meta_start = TimeStart();
        // Default request is 3000 before and after, which should be enough, exact amount is calculated later based
        // on the data. It would not influence performance on GPU anyway.
        size_t packets_before_sensing_start{3000};
        size_t packets_after_sensing_stop{3000};
        // This will make sure that sequence starts with an echo packet in case of envisat.
        const auto packets_fetch_meta =
            alus::asar::envformat::FetchMeta(l0_ds, asar_meta.sensing_start, asar_meta.sensing_stop, product_type,
                                             packets_before_sensing_start, packets_after_sensing_stop);
        // Arbitrary minimum is chosen as 100.
        if (packets_fetch_meta.size() < 100) {
            throw std::runtime_error("Only " + std::to_string(packets_fetch_meta.size()) +
                                     " packets fetched, please check the dataset or sensing parameters.");
        }
        TimeStop(fetch_meta_start, "Metadata forecast fetch ");

        const auto packet_fetch_start = TimeStart();
        const auto& mdsr = alus::asar::envformat::ParseSphAndGetMdsr(asar_meta, sar_meta, l0_ds);
        std::vector<alus::asar::envformat::CommonPacketMetadata> packets_metadata(packets_fetch_meta.size());
        LOGD << "Parsing measurements from " << MjdToPtime(packets_fetch_meta.front().isp_sensing_time) << " until "
             << MjdToPtime(packets_fetch_meta.back().isp_sensing_time);
        LOGD << "That makes " << packets_before_sensing_start << " packets before and " << packets_after_sensing_stop
             << " packets after the assigned sensing start and stop";
        // Next level would be to allocate RAM buffer array async ahead - even if too much, but on average basis based
        // on sensing period and mode.
        const auto compressed_measurements = alus::asar::envformat::ParseLevel0Packets(
            l0_ds, mdsr.ds_offset, packets_fetch_meta, product_type, ins_file, packets_metadata);

        alus::asar::mainflow::AssembleMetadataFrom(packets_metadata, asar_meta, sar_meta, ins_file,
                                                   compressed_measurements.max_samples,
                                                   compressed_measurements.total_samples, product_type);
        // This is open issue - what exactly constitutes to product error.
        // Currently only when ERS has missing packets base on data record no.
        if (compressed_measurements.no_of_product_errors_compensated > 0) {
            asar_meta.product_err = true;
        }

        while (!cuda_init.IsFinished())
            ;
        cuda_init.CheckErrors();
        cufftComplex* d_parsed_packets;
        alus::asar::mainflow::ConvertRawSampleSetsToComplex(compressed_measurements, packets_metadata, sar_meta,
                                                            ins_file, product_type, &d_parsed_packets);

        TimeStop(packet_fetch_start, "Packets and sar_meta parsing plus converting to complex samples");

        LOGD << "Total fetched meta for forecast - " << packets_fetch_meta.size();
        LOGD << "Total parsed L0 packet - " << sar_meta.img.azimuth_size;

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

        const int az_size = sar_meta.img.azimuth_size;
        const int rg_size = sar_meta.img.range_size;
        const auto fft_sizes = GetOptimalFFTSizes();
        LOGD << "Starting with " << sar_meta.img.range_size << " range samples";

        const auto rg_plan_sz = fft_sizes.upper_bound(rg_size + sar_meta.chirp.n_samples);
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

        CHECK_CUDA_ERR(cudaMemcpy2D(img.Data(), rg_padded * 8, d_parsed_packets, rg_size * 8, rg_size * 8, az_size,
                                    cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERR(cudaFree(d_parsed_packets));

        LOGD << "Continuing with " << img.XSize() << " range samples" << sar_meta.img.range_size;

        TimeStop(gpu_transfer_start, "GPU image formation");

        auto correction_start = TimeStart();
        RawDataCorrection(img, {sar_meta.total_raw_samples}, sar_meta.results);
        img.ZeroNaNs();
        TimeStop(correction_start, "I/Q correction");

        LOGD << "After raw data correction " << img.XSize() << " range samples " << sar_meta.img.range_size;

        auto vr_start = TimeStart();
        // TODO - Would give exactly the same results as in EstimateProcessingVelocity()'s sub step.
        // Results would change after VR_poly and doppler_centroid_poly are updated ?!
        auto aperture_pixels = CalcAperturePixels(sar_meta);
        sar_meta.results.Vr_poly = EstimateProcessingVelocity(sar_meta);
        TimeStop(vr_start, "Vr estimation");

        auto chirp = GenerateChirpData(sar_meta.chirp, rg_padded);

        auto dc_start = TimeStart();
        // Element from the back() indicates polynomial starting value.
        sar_meta.results.doppler_centroid_poly = CalculateDopplerCentroid(img, sar_meta.pulse_repetition_frequency);
        TimeStop(dc_start, "fractional DC estimation");

        if (args.StorePlots()) {
            alus::asar::mainflow::StorePlots(args.GetOutputPath().data(), wif_name_base, sar_meta, chirp);
        }

        if (args.StoreIntensity()) {
            alus::asar::mainflow::StoreIntensity(args.GetOutputPath().data(), wif_name_base, "raw", img);
        }

        LOGD << "Image GPU byte size = " << (img.TotalByteSize() / 1e9) << "GB";
        LOGD << "Estimated GPU memory usage ~" << ((img.TotalByteSize() * 2) / 1e9) << "GB";

        CudaWorkspace d_workspace(img.TotalByteSize());

        auto rc_start = TimeStart();
        RangeCompression(img, chirp, sar_meta.chirp.n_samples, d_workspace);
        TimeStop(rc_start, "Range compression");

        LOGD << "After range compression " << img.XSize() << " range samples " << sar_meta.img.range_size;

        sar_meta.img.range_size = img.XSize();

        if (args.StoreIntensity()) {
            alus::asar::mainflow::StoreIntensity(args.GetOutputPath().data(), wif_name_base, "rc", img);
        }

        auto az_comp_start = TimeStart();
        DevicePaddedImage az_compressed_image;
        RangeDopplerAlgorithm(sar_meta, img, az_compressed_image, d_workspace);
        // img is unused from now on, everything got traded to 'az_compressed_image'.

        az_compressed_image.ZeroFillPaddings();

        std::string lvl1_out_name = asar_meta.product_name;
        lvl1_out_name.at(6) = 'S';
        lvl1_out_name.at(8) = '1';
        const std::string lvl1_out_full_path =
            std::string(args.GetOutputPath()) + std::filesystem::path::preferred_separator + lvl1_out_name;
        auto lvl1_file_handle = alus::util::FileAsync(lvl1_out_full_path);

        const auto az_rg_windowing = alus::asar::mainflow::CalcResultsWindow(
            sar_meta, packets_before_sensing_start, packets_after_sensing_stop,
            *std::max_element(aperture_pixels.cbegin(), aperture_pixels.cend()));

        CHECK_CUDA_ERR(cudaDeviceSynchronize());  // For ZeroFillPaddings() - the kernel was left running async.
        TimeStop(az_comp_start, "Azimuth compression");

        if (args.StoreIntensity()) {
            alus::asar::mainflow::StoreIntensity(args.GetOutputPath().data(), wif_name_base, "proc_slc",
                                                 az_compressed_image);
        }

        auto result_file_assembly = TimeStart();

        DevicePaddedImage subsetted_raster;
        alus::asar::mainflow::SubsetResultsAndReassembleMeta(az_compressed_image, az_rg_windowing, packets_metadata,
                                                             asar_meta, sar_meta, ins_file, product_type, d_workspace,
                                                             subsetted_raster);
        // az_compressed_image is unused from now on.
        sar_meta.img.range_size = subsetted_raster.XSize(); // TODO - hackathon
        constexpr size_t record_header_bytes = 12 + 1 + 4;
        const auto mds_record_size = subsetted_raster.XSize() * sizeof(IQ16) + record_header_bytes;
        const auto mds_record_count = subsetted_raster.YSize();
        MDS mds;
        mds.n_records = mds_record_count;
        mds.record_size = mds_record_size;
        auto mds_buffer_init = std::async(std::launch::async, [&mds] {
            mds.buf = static_cast<char*>(alus::util::Memalloc(mds.n_records * mds.record_size));
        });

        EnvisatIMS ims{};
        ConstructIMS(ims, sar_meta, asar_meta, mds, GetSoftwareVersion());
        if (!lvl1_file_handle.CanWrite(std::chrono::seconds(10))) {
            throw std::runtime_error("Could not create L1 file at " + lvl1_out_full_path);
        }
        lvl1_file_handle.Write(&ims, sizeof(ims));

        TimeStop(result_file_assembly, "Creating IMS and write start, subsetting data");

        auto result_correction_start = TimeStart();
        if (d_workspace.ByteSize() < static_cast<size_t>(mds.record_size * mds.n_records)) {
            throw std::logic_error(
                "Implementation error occurred - CUDA workspace buffer shall be made larger or equal to what is "
                "required for MDS buffer.");
        }

        float tambov{120000 / 100};
        //        const auto swath_idx = alus::asar::envformat::SwathIdx(asar_meta.swath);
        //        const auto tambov = xca.scaling_factor_im_slc_vv[swath_idx];
        auto device_mds_buf = d_workspace.GetAs<char>();
        alus::asar::mainflow::FormatResults(subsetted_raster, device_mds_buf, record_header_bytes, tambov);
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
