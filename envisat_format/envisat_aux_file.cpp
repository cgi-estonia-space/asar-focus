

#include "envisat_aux_file.h"

#include "util/checks.h"
#include "util/date_time_util.h"

// TODO functions are pretty much copy-paste, refactor it?

void FindCONFile(std::string aux_root, boost::posix_time::ptime start, ConfigurationFile &conf_file,
                 std::string &filename) {
    std::filesystem::path ins_dir(aux_root);
    ins_dir /= "ASA_CON_AX";

    // ASA_CON_AXVIEC20030909_000000_20020815_000000_20021017_130000

    // sizes: mph 1247, sph 378

    constexpr size_t CONF_FILE_SIZE = 5721;
    constexpr size_t MPH_SPH_SIZE = 1247 + 378;
    static_assert(CONF_FILE_SIZE == sizeof(conf_file) + MPH_SPH_SIZE);

    std::vector<uint8_t> conf_data(CONF_FILE_SIZE);

    bool ok = false;
    for (auto const &dir_entry: std::filesystem::directory_iterator(ins_dir)) {
        auto fn = dir_entry.path().stem().string();
        std::string start_date = fn.substr(30, 8);
        std::string end_date = fn.substr(46, 8);

        if (start >= alus::util::date_time::YYYYMMDD(start_date) && start < alus::util::date_time::YYYYMMDD(end_date)) {
            std::cout << "Configuration file = " << fn << "\n";
            filename = fn;

            FILE *fp = fopen(dir_entry.path().c_str(), "r");
            auto total = fread(conf_data.data(), 1, CONF_FILE_SIZE, fp);

            if (total != CONF_FILE_SIZE) {
                ERROR_EXIT("Configuration file incorrect size?");
            }
            fclose(fp);

            ok = true;

            break;
        }
    }

    if (!ok) {
        ERROR_EXIT("No configuration file found!?");
    }

    conf_data.erase(conf_data.begin(), conf_data.begin() + MPH_SPH_SIZE);
    memcpy(&conf_file, conf_data.data(), conf_data.size());
    conf_file.BSwap();
}

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

        if (start >= alus::util::date_time::YYYYMMDD(start_date) && start < alus::util::date_time::YYYYMMDD(end_date)) {
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
