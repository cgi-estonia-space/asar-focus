/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "main_flow.h"

#include <optional>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "alus_log.h"
#include "date_time_util.h"
#include "envisat_format_kernels.h"
#include "ers_aux_file.h"
#include "img_output.h"
#include "plot.h"
#include "sar/iq_correction.cuh"
#include "status_assembly.h"

namespace alus::asar::mainflow {

void CheckAndLimitSensingStartEnd(boost::posix_time::ptime& product_sensing_start,
                                  boost::posix_time::ptime& product_sensing_stop,
                                  std::optional<std::string_view> user_sensing_start,
                                  std::optional<std::string_view> user_sensing_end) {
    if (!user_sensing_start.has_value()) {
        return;
    }

    const auto user_start = alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff(std::string(user_sensing_start.value()));
    if (user_start < product_sensing_start) {
        LOGE << "User specified sensing start '" << boost::posix_time::to_simple_string(user_start)
             << "' is earlier than product's packets - " << boost::posix_time::to_simple_string(product_sensing_start);
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    if (user_start > product_sensing_stop) {
        LOGE << "User specified sensing start '" << boost::posix_time::to_simple_string(user_start)
             << "' is later than product's sensing end time - "
             << boost::posix_time::to_simple_string(product_sensing_stop);
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    product_sensing_start = user_start;

    if (!user_sensing_end.has_value()) {
        return;
    }

    const auto user_end = alus::util::date_time::ToYyyyMmDd_HhMmSsFfffff(std::string(user_sensing_end.value()));
    if (user_end < user_start) {
        LOGE << "Invalid sensing end time specified, because it is earlier than start.";
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    if (user_end > product_sensing_stop) {
        LOGE << "User specified sensing end '" << boost::posix_time::to_simple_string(user_end)
             << "' is later than product's sensing end time - "
             << boost::posix_time::to_simple_string(product_sensing_stop);
        exit(alus::asar::status::EXIT_CODE::INVALID_ARGUMENT);
    }

    product_sensing_stop = user_end;
}

specification::ProductTypes TryDetermineProductType(std::string_view product_name) {
    alus::asar::specification::ProductTypes product_type = alus::asar::specification::GetProductTypeFrom(product_name);
    if (product_type == alus::asar::specification::ProductTypes::UNIDENTIFIED) {
        LOGE << "This product is not supported - " << product_name;
        exit(alus::asar::status::EXIT_CODE::UNSUPPORTED_DATASET);
    }

    return product_type;
}

void TryFetchOrbit(alus::dorisorbit::Parsable& orbit_source, ASARMetadata& asar_meta, SARMetadata& sar_meta) {
    sar_meta.osv = orbit_source.CreateOrbitInfo(asar_meta.sensing_start, asar_meta.sensing_stop);

    if (sar_meta.osv.size() < 8) {
        LOGE << "Could not fetch enough orbit state vectors from input '"
             << orbit_source.GetL1ProductMetadata().orbit_name << "' atleast 8 required.";
        if (sar_meta.osv.size() > 0) {
            LOGE << "First OSV date - " << boost::posix_time::to_simple_string(sar_meta.osv.front().time);
            LOGE << "Last OSV date - " << boost::posix_time::to_simple_string(sar_meta.osv.back().time);
        }
        exit(alus::asar::status::EXIT_CODE::INVALID_DATASET);
    }
}

void FetchAuxFiles(InstrumentFile& ins_file, ConfigurationFile& conf_file, ASARMetadata& asar_meta,
                   specification::ProductTypes product_type, std::string_view aux_path) {
    if (product_type == alus::asar::specification::ProductTypes::ASA_IM0) {
        FindINSFile(std::string(aux_path), asar_meta.sensing_start, ins_file, asar_meta.instrument_file);
        FindCONFile(std::string(aux_path), asar_meta.sensing_start, conf_file, asar_meta.configuration_file);
    } else if (product_type == alus::asar::specification::ProductTypes::SAR_IM0) {
        alus::asar::envisat_format::FindINSFile(std::string(aux_path), asar_meta.sensing_start, ins_file,
                                                asar_meta.instrument_file);
        alus::asar::envisat_format::FindCONFile(std::string(aux_path), asar_meta.sensing_start, conf_file,
                                                asar_meta.configuration_file);
    } else {
        LOGE << "Implementation error for this type of product while trying to fetch auxiliary files";
        exit(alus::asar::status::EXIT_CODE::ASSERT_FAILURE);
    }
}

void FormatResults(DevicePaddedImage& img, char* dest_space, size_t record_header_size, float calibration_constant) {
    envformat::ConditionResults(img, dest_space, record_header_size, calibration_constant);
}

void StorePlots(std::string output_path, std::string product_name, const SARMetadata& sar_metadata,
                const std::vector<std::complex<float>>& chirp) {
    {
        PlotArgs plot_args = {};
        plot_args.out_path = output_path + "/" + product_name + "_Vr.html";
        plot_args.x_axis_title = "range index";
        plot_args.y_axis_title = "Vr(m/s)";
        plot_args.data.resize(1);
        auto& line = plot_args.data[0];
        line.line_name = "Vr";
        PolyvalRange(sar_metadata.results.Vr_poly, 0, sar_metadata.img.range_size, line.x, line.y);
        Plot(plot_args);
    }
    {
        PlotArgs plot_args = {};
        plot_args.out_path = output_path + "/" + product_name + "_chirp.html";
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
    {
        PlotArgs plot_args = {};
        plot_args.out_path = output_path + "/" + product_name + "_dc.html";
        plot_args.x_axis_title = "range index";
        plot_args.y_axis_title = "Hz";
        plot_args.data.resize(1);
        auto& line = plot_args.data[0];
        line.line_name = "Doppler centroid";
        PolyvalRange(sar_metadata.results.doppler_centroid_poly, 0, sar_metadata.img.range_size, line.x, line.y);
        Plot(plot_args);
    }
}

void StoreIntensity(std::string output_path, std::string product_name, std::string postfix,
                    const DevicePaddedImage& dev_padded_img) {
    std::string path = output_path + "/";
    path += product_name + "_" + postfix + ".tif";
    WriteIntensityPaddedImg(dev_padded_img, path.c_str());
}

}  // namespace alus::asar::mainflow