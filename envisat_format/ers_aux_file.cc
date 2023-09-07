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

#include "util/date_time_util.h"

namespace alus::asar::envisat_format {

    void FindINSFile(std::string aux_root, boost::posix_time::ptime start, InstrumentFile &ins_file,
                     std::string &filename) {
        std::filesystem::path ins_dir(aux_root);
        ins_dir /= "ASA_INS_AX";

        if (!std::filesystem::exists(ins_dir)) {
            std::cerr << "There is no '" << "ASA_INS_AX" << "' inside auxiliary folder - " << aux_root << std::endl;
            exit(1);
        }

        // ASA_INS_AXVIEC20020308_112323_20020301_000000_20021231_000000

        // sizes: mph 1247, sph 378

        constexpr size_t INS_FILE_SIZE = 173273;
        constexpr size_t MPH_SPH_SIZE = 1247 + 378;
        static_assert(INS_FILE_SIZE == sizeof(ins_file) + MPH_SPH_SIZE);

        std::vector<uint8_t> ins_data(INS_FILE_SIZE);

        bool ok = false;
        for (auto const &dir_entry: std::filesystem::directory_iterator(ins_dir)) {
            auto fn = dir_entry.path().stem().string();
            std::string start_date = fn.substr(30, 8);
            std::string end_date = fn.substr(46, 8);

            if (start >= alus::util::date_time::YYYYMMDD(start_date) &&
                start < alus::util::date_time::YYYYMMDD(end_date)) {
                std::cout << "Instrument file = " << fn << "\n";
                filename = fn;

                FILE *fp = fopen(dir_entry.path().c_str(), "r");
                auto total = fread(ins_data.data(), 1, INS_FILE_SIZE, fp);

                if (total != INS_FILE_SIZE) {
                    ERROR_EXIT("Instrument file incorrect size?");
                }
                fclose(fp);

                ok = true;

                break;
            }
        }

        if (!ok) {
            ERROR_EXIT("No instruments file found!?");
        }

        ins_data.erase(ins_data.begin(), ins_data.begin() + MPH_SPH_SIZE);

        memcpy(&ins_file, ins_data.data(), ins_data.size());
        ins_file.BSwap();

    }
}