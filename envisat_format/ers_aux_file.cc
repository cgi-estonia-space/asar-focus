/**
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
*
* ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
* Creative Commons Attribution-ShareAlike 4.0 International License.
*
* You should have received a copy of the license along with this
* work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
*/
#include "envisat_format/ers_aux_file.h"

#include <string>

#include <boost/date_time/posix_time/ptime.hpp>

#include "asar_aux.h"
#include "checks.h"

namespace {
    std::string DetermineFilePath(std::string aux_root, boost::posix_time::ptime start, alus::asar::aux::Type t) {
        if (!std::filesystem::exists(aux_root)) {
            ERROR_EXIT("The auxiliary folder - " + aux_root + " - does not exist.");
        }

        // ASA_CON_AXVIEC20030909_000000_20020815_000000_20021017_130000
        const auto file_path = alus::asar::aux::GetPathFrom(aux_root, start, t);
        if (file_path.empty()) {
            ERROR_EXIT("Could not find aux file for ENVISAT in " + aux_root);
        }

        return file_path;
    }
}

namespace alus::asar::envisat_format {

    // Generalize this auxiliary file find code.
    void FindINSFile(std::string aux_root, boost::posix_time::ptime start, InstrumentFile &ins_file,
                     std::string &filename) {

        // ER1_INS_AXVXXX20110104_154739_19910717_000000_20000310_000000
        const auto file_path = DetermineFilePath(aux_root, start, alus::asar::aux::Type::INSTRUMENT_CHARACTERIZATION);

        // sizes: mph 1247, sph 378
        constexpr size_t INS_FILE_SIZE = 173273;
        constexpr size_t MPH_SPH_SIZE = 1247 + 378;
        static_assert(INS_FILE_SIZE == sizeof(ins_file) + MPH_SPH_SIZE);

        std::vector<uint8_t> ins_data(INS_FILE_SIZE);
        FILE *fp = fopen(file_path.c_str(), "r");
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


    void FindCONFile(std::string aux_root, boost::posix_time::ptime start, ConfigurationFile& conf_file,
                     std::string& filename) {

        const auto file_path = DetermineFilePath(aux_root, start, alus::asar::aux::Type::PROCESSOR_CONFIGURATION);

        // sizes: mph 1247, sph 378
        constexpr size_t CONF_FILE_SIZE = 5721;
        constexpr size_t MPH_SPH_SIZE = 1247 + 378;
        static_assert(CONF_FILE_SIZE == sizeof(conf_file) + MPH_SPH_SIZE);

        std::vector<uint8_t> conf_data(CONF_FILE_SIZE);
        FILE *fp = fopen(file_path.c_str(), "r");
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
}