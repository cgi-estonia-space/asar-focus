

#include "envisat_ins_file.h"

#include "util/checks.h"

namespace {
boost::posix_time::ptime YYYYMMDD(std::string str) {
    auto* facet = new boost::posix_time::time_input_facet("%Y%m%d");
    std::stringstream date_stream(str);
    date_stream.imbue(std::locale(std::locale::classic(), facet));
    boost::posix_time::ptime time;
    date_stream >> time;
    return time;
}

}  // namespace

void FindInsFile(std::string path, boost::posix_time::ptime start, InstrumentFile& ins_file, std::string& filename) {
    // ASA_INS_AXVIEC20020308_112323_20020301_000000_20021231_000000

    // sizes: mph 1247, sph 378

    constexpr size_t INS_FILE_SIZE = 173273;

    std::vector<uint8_t> ins_data(INS_FILE_SIZE);

    bool ok = false;
    for (auto const& dir_entry : std::filesystem::directory_iterator(path)) {
        auto fn = dir_entry.path().stem().string();
        std::cout << fn << "\n";
        std::string start_date = fn.substr(30, 8);
        std::string end_date = fn.substr(46, 8);

        if (start >= YYYYMMDD(start_date) && start < YYYYMMDD(end_date)) {
            std::cout << "Instrument file = " << fn;
            filename = fn;

            FILE* fp = fopen(dir_entry.path().c_str(), "r");
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

    ins_data.erase(ins_data.begin(), ins_data.begin() + (1247 + 378));
    // FILE* fp = fopen("/tmp/ins_no_h.txt", "w");
    // fwrite(ins_data.data(), 1, ins_data.size(), fp);
    // fclose(fp);

    memcpy(&ins_file, ins_data.data(), ins_data.size());
    ins_file.BSwap();
}