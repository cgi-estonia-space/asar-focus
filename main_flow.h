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
#include "patc.h"
#include "sar/focussing_details.h"
#include "sar/sar_metadata.h"

namespace alus::asar::mainflow {

void CheckAndLimitSensingStartEnd(boost::posix_time::ptime& product_sensing_start,
                                  boost::posix_time::ptime& product_sensing_stop,
                                  std::optional<std::string_view> user_sensing_start,
                                  std::optional<std::string_view> user_sensing_end);
specification::ProductTypes TryDetermineProductType(std::string_view product_name);

void TryFetchOrbit(alus::dorisorbit::Parsable& orbit_source, ASARMetadata& asar_meta, SARMetadata& sar_meta);

void FetchAuxFiles(InstrumentFile& ins_file, ConfigurationFile& conf_file, envformat::aux::ExternalCalibration& xca,
                   std::optional<envformat::aux::Patc>& patc, ASARMetadata& asar_meta,
                   specification::ProductTypes product_type, std::string_view aux_path);

struct AzimuthRangeWindow {
    size_t near_range_pixels_to_remove;
    size_t far_range_pixels_to_remove;
    size_t lines_to_remove_before_sensing_start;
    size_t lines_to_remove_after_sensing_start;
    size_t lines_to_remove_before_sensing_stop;
    size_t lines_to_remove_after_sensing_stop;
};

AzimuthRangeWindow CalcResultsWindow(double doppler_centroid_constant_term, size_t lines_before_sensing,
                                     size_t lines_after_sensing, size_t max_aperture_pixels,
                                     const sar::focus::RcmcParameters& rcmc_params);

// Calibrate, clamp and Big endian results with empty header into dest_space, which shall be a device memory.
void FormatResultsSLC(DevicePaddedImage& img, char* dest_space, size_t record_header_size, float calibration_constant);
void FormatResultsDetected(const float* d_img, char* dest_space, int range_size, int azimuth_size,
                           size_t record_header_size, float calibration_constant);

void StorePlots(std::string output_path, std::string product_name, const SARMetadata& sar_metadata);

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
                          size_t range_pixels_recalib = 0);

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

void PrefillIms(EnvisatSubFiles& ims, size_t total_packets_processed);

void CheckAndRegisterPatc(const envformat::aux::Patc& patc, ASARMetadata& metadata);

// I/Q mean I/Q std dev
std::array<double, 4> CalculateStatistics(const DevicePaddedImage& img);

// Amplitude mean, 0, Amplitude std dev, 0
std::array<double, 4> CalculateStatisticsImaged(const DevicePaddedImage& img);

}  // namespace alus::asar::mainflow
