/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */
#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "asar_constants.h"
#include "cuda_workplace.h"
#include "device_padded_image.cuh"
#include "doris_orbit.h"
#include "envisat_aux_file.h"
#include "envisat_lvl0_parser.h"
#include "envisat_lvl1_writer.h"
#include "envisat_types.h"
#include "sar/sar_metadata.h"

namespace alus::asar::mainflow {

void CheckAndLimitSensingStartEnd(boost::posix_time::ptime& product_sensing_start,
                                  boost::posix_time::ptime& product_sensing_stop,
                                  std::optional<std::string_view> user_sensing_start,
                                  std::optional<std::string_view> user_sensing_end);
specification::ProductTypes TryDetermineProductType(std::string_view product_name);

void TryFetchOrbit(alus::dorisorbit::Parsable& orbit_source, ASARMetadata& asar_meta, SARMetadata& sar_meta);

void FetchAuxFiles(InstrumentFile& ins_file, ConfigurationFile& conf_file, envformat::aux::ExternalCalibration& xca,
                   ASARMetadata& asar_meta, specification::ProductTypes product_type, std::string_view aux_path);

struct AzimuthRangeWindow {
    size_t near_range_pixels_to_remove;
    size_t far_range_pixels_to_remove;
    size_t lines_to_remove_before_sensing_start;
    size_t lines_to_remove_after_sensing_start;
    size_t lines_to_remove_before_sensing_stop;
    size_t lines_to_remove_after_sensing_stop;
};

AzimuthRangeWindow CalcResultsWindow(const SARMetadata& sar_meta, size_t lines_before_sensing,
                                     size_t lines_after_sensing, size_t max_aperture_pixels);

// Calibrate, clamp and Big endian results with empty header into dest_space, which shall be a device memory.
void FormatResults(DevicePaddedImage& img, char* dest_space, size_t record_header_size, float calibration_constant);

void StorePlots(std::string output_path, std::string product_name, const SARMetadata& sar_metadata,
                const std::vector<std::complex<float>>& chirp);

void StoreIntensity(std::string output_path, std::string product_name, std::string postfix,
                    const DevicePaddedImage& dev_padded_img);

/*
 * Can be called multiple times during processing. In order to separate calculations before compression the SWST
 * multiplier can be left unspecified e.g. default -1. After (range) compressions SWST multiplier can be set to 0 in
 * order to not screw up sample count calculation because SWST influenced offsets and everything has been
 * compensated/removed.
 */
void AssembleMetadataFrom(std::vector<envformat::CommonPacketMetadata>& parsed_meta, ASARMetadata& asar_meta,
                          SARMetadata& sar_meta, InstrumentFile& ins_file, size_t max_raw_samples_at_range,
                          size_t total_raw_samples, alus::asar::specification::ProductTypes product_type,
                          int swst_multiplier = -1);

// d_converted_measurements could be CudaWorkspace array now since we can estimate FFT beforehand when parsing metadata
// separately now.
void ConvertRawSampleSetsToComplex(const envformat::RawSampleMeasurements& raw_measurements,
                                   const std::vector<envformat::CommonPacketMetadata>& parsed_meta,
                                   const SARMetadata& sar_meta, const InstrumentFile& ins_file,
                                   specification::ProductTypes product_type, cufftComplex** d_converted_measurements);

void SubsetResultsAndReassembleMeta(DevicePaddedImage& azimuth_compressed_raster, AzimuthRangeWindow window,
                                    std::vector<alus::asar::envformat::CommonPacketMetadata>& common_metadata,
                                    ASARMetadata& asar_meta, SARMetadata& sar_meta, InstrumentFile& ins_file,
                                    alus::asar::specification::ProductTypes product_type, CudaWorkspace& workspace,
                                    DevicePaddedImage& subsetted_raster);

void PrefillIms(EnvisatIMS& ims, size_t total_packets_processed);

}  // namespace alus::asar::mainflow
