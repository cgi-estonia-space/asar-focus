/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "envisat_aux_file.h"
#include "alus_log.h"
#include "asar_aux.h"
#include "checks.h"
#include "filesystem_util.h"
#include "patc.h"

namespace {
std::string DetermineFilePath(std::string aux_root, boost::posix_time::ptime start, alus::asar::aux::Type t) {
    if (!std::filesystem::exists(aux_root)) {
        throw std::runtime_error("The auxiliary folder - " + aux_root + " - does not exist.");
    }

    // ASA_CON_AXVIEC20030909_000000_20020815_000000_20021017_130000
    const auto file_path = alus::asar::aux::GetPathFrom(aux_root, start, t);
    if (file_path.empty()) {
        throw std::runtime_error("Could not find aux file in " + aux_root);
    }

    return file_path;
}
}  // namespace

void FindCONFile(std::string aux_root, boost::posix_time::ptime start, ConfigurationFile& conf_file,
                 std::string& filename) {
    const auto file_path = DetermineFilePath(aux_root, start, alus::asar::aux::Type::PROCESSOR_CONFIGURATION);

    // sizes: mph 1247, sph 378
    constexpr size_t CONF_FILE_SIZE = 5721;
    constexpr size_t MPH_SPH_SIZE = 1247 + 378;
    static_assert(CONF_FILE_SIZE == sizeof(conf_file) + MPH_SPH_SIZE);

    std::vector<uint8_t> conf_data(CONF_FILE_SIZE);

    LOGD << "Processor configuration file = " << file_path;
    FILE* fp = fopen(file_path.c_str(), "r");
    auto total = fread(conf_data.data(), 1, CONF_FILE_SIZE, fp);

    if (total != CONF_FILE_SIZE) {
        ERROR_EXIT("Configuration file incorrect size?");
    }
    fclose(fp);

    conf_data.erase(conf_data.begin(), conf_data.begin() + MPH_SPH_SIZE);
    memcpy(&conf_file, conf_data.data(), conf_data.size());
    conf_file.BSwap();
    filename = std::filesystem::path(file_path).filename();
}

void FindINSFile(std::string aux_root, boost::posix_time::ptime start, InstrumentFile& ins_file,
                 std::string& filename) {
    const auto file_path = DetermineFilePath(aux_root, start, alus::asar::aux::Type::INSTRUMENT_CHARACTERIZATION);
    // ASA_INS_AXVIEC20020308_112323_20020301_000000_20021231_000000

    // sizes: mph 1247, sph 378

    constexpr size_t INS_FILE_SIZE = 173273;
    constexpr size_t MPH_SPH_SIZE = 1247 + 378;
    static_assert(INS_FILE_SIZE == sizeof(ins_file) + MPH_SPH_SIZE);

    std::vector<uint8_t> ins_data(INS_FILE_SIZE);

    LOGD << "Instrument file = " << file_path;
    FILE* fp = fopen(file_path.c_str(), "r");
    auto total = fread(ins_data.data(), 1, INS_FILE_SIZE, fp);

    if (total != INS_FILE_SIZE) {
        ERROR_EXIT("Instrument file incorrect size?");
    }
    fclose(fp);

    ins_data.erase(ins_data.begin(), ins_data.begin() + MPH_SPH_SIZE);

    memcpy(&ins_file, ins_data.data(), ins_data.size());
    ins_file.BSwap();
    filename = std::filesystem::path(file_path).filename();
}

namespace alus::asar::envformat::aux {

void GetXca(std::string_view aux_root, boost::posix_time::ptime start, ExternalCalibration& xca,
            std::string& filename) {
    const auto file_path = DetermineFilePath(aux_root.data(), start, alus::asar::aux::Type::EXTERNAL_CALIBRATION);

    constexpr size_t MPH_SIZE = 1247;
    constexpr size_t SPH_SIZE = 378;
    constexpr size_t GADS_SIZE = 26552;
    constexpr size_t XCA_FILE_SIZE = MPH_SIZE + SPH_SIZE + GADS_SIZE;
    static_assert(GADS_SIZE == sizeof(ExternalCalibration));
    static_assert(XCA_FILE_SIZE == 28177);

    std::vector<uint8_t> file_contents(XCA_FILE_SIZE);

    LOGD << "External calibration file = " << file_path;
    FILE* fp = fopen(file_path.c_str(), "r");
    auto total = fread(file_contents.data(), 1, XCA_FILE_SIZE, fp);

    if (total != XCA_FILE_SIZE) {
        throw std::runtime_error("Did not manage to read " + std::to_string(XCA_FILE_SIZE) + " bytes from " +
                                 file_path + ". Please check the file.");
    }
    fclose(fp);

    file_contents.erase(file_contents.begin(), file_contents.begin() + MPH_SIZE + SPH_SIZE);

    memcpy(&xca, file_contents.data(), file_contents.size());
    xca.Bswap();

    // It has been observed to be zero for ERS XCA files.
    if (xca.dsr_length != GADS_SIZE && xca.dsr_length != 0) {
        throw std::runtime_error("External calibration file - " + file_path + " - GADS DSR length " +
                                 std::to_string(xca.dsr_length) + " bytes is not correct, should be " +
                                 std::to_string(GADS_SIZE));
    }

    filename = std::filesystem::path(file_path).filename();
}

std::optional<Patc> GetTimeCorrelation(std::string_view aux_root) {
    const std::regex patc_reg{R"(PATC.*)"};

    if (!std::filesystem::exists(aux_root)) {
        throw std::runtime_error("The auxiliary folder - " + std::string(aux_root) + " - does not exist.");
    }

    const auto patc_listing = util::filesystem::GetFileListingAt(aux_root, patc_reg);
    if (patc_listing.size() > 1) {
        throw std::runtime_error("Expected single PATC file in the '" + std::string(aux_root) + "'. There were " +
                                 std::to_string(patc_listing.size()) + " found.");
    }

    if (patc_listing.size() == 0) {
        LOGW << "No PATC files found at " << aux_root;
        return std::nullopt;
    }

    return ParsePatc(patc_listing.front());
}

}  // namespace alus::asar::envformat::aux
