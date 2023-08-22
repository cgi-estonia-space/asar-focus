

#include "envisat_file_format.h"

#include "asar_lvl0_parser.h"
//#include "envisat_utils.h"
#include "envisat_ph.h"
#include "envisat_ins_file.h"

#include "gdal/gdal_priv.h"
#include "orbit_state_vector.h"



template<class T>
[[nodiscard]]
const uint8_t* CopyBSwapPOD(T& dest, const uint8_t* src)
{
    memcpy(&dest, src, sizeof(T));
    dest = bswap(dest);
    return src + sizeof(T);
}


struct EchoMeta
{
    mjd isp_sensing_time;
    uint8_t mode_id;
    uint64_t onboard_time;
    uint32_t mode_count;
    uint8_t antenna_beam_set_no;
    uint8_t comp_ratio;
    bool echo_flag;
    bool noise_flag;
    bool cal_flag;
    bool cal_type;
    uint16_t cycle_count;

    uint16_t pri_code;
    uint16_t swst_code;
    uint16_t echo_window_code;
    uint8_t upconverter_raw;
    uint8_t downconverter_raw;
    bool tx_pol;
    bool rx_pol;
    uint8_t cal_row_number;
    uint16_t tx_pulse_code;
    uint8_t beam_adj_delta;
    uint8_t chirp_pulse_bw_code;
    uint8_t aux_tx_monitor_level;
    uint16_t resampling_factor;


    std::vector<std::complex<float>> raw_data;
};


int FBAQ4Idx(int block, int idx)
{

    switch(idx)
    {
    case 0b1111:
        idx = 0;
        break;
    case 0b1110:
        idx = 1;
        break;
    case 0b1101:
        idx = 2;
        break;
    case 0b1100:
        idx = 3;
        break;
    case 0b1011:
        idx = 4;
        break;
    case 0b1010:
        idx = 5;
        break;
    case 0b1001:
        idx = 6;
        break;
    case 0b1000:
        idx = 7;
        break;
    case 0b0000:
        idx = 8;
        break;
    case 0b0001:
        idx = 9;
        break;
    case 0b0010:
        idx = 10;
        break;
    case 0b0011:
        idx = 11;
        break;
    case 0b0100:
        idx = 12;
        break;
    case 0b0101:
        idx = 13;
        break;
    case 0b0110:
        idx = 14;
        break;
    case 0b0111:
        idx = 15;
        break;
    }

    return 256 * idx + block;
}

double NadirLLParse(const std::string& str)
{
    auto val = std::stol(str.substr(3, 8));
    if(str.at(0) == '-')
    {
        val = -val;
    }
    return val * 1e-6;
}

void ParseIMFile(const std::vector<char>& file_data, const char* aux_path, SARMetadata& sar_meta, ASARMetadata& asar_meta, std::vector<std::complex<float>>& img_data)
{
    ProductHeader mph = {};

    mph.Load(0, file_data.data(), MPH_SIZE);

    printf("MPH = \n");
    mph.PrintValues();

    asar_meta.product_name = mph.Get("PRODUCT");

    if(!boost::algorithm::starts_with(asar_meta.product_name, "ASA_IM__0"))
    {
        ERROR_EXIT("Envisat IM lvl 0 files only at this point!");
    }


    asar_meta.sensing_start = StrToPtime(mph.Get("SENSING_START"));
    asar_meta.sensing_stop = StrToPtime(mph.Get("SENSING_STOP"));

    asar_meta.first_line_time = asar_meta.sensing_start;
    asar_meta.last_line_time = asar_meta.sensing_stop;


    sar_meta.osv = FindOrbits(asar_meta.sensing_start, asar_meta.sensing_stop);
    sar_meta.platform_velocity = CalcVelocity(sar_meta.osv[sar_meta.osv.size()/2]);
    sar_meta.results.Vr = sar_meta.platform_velocity * 0.94;

    printf("platform velocity = %f, initial Vr = %f\n", sar_meta.platform_velocity, sar_meta.results.Vr);


    InstrumentFile ins_file = {};
    FindInsFile(aux_path, asar_meta.sensing_start, ins_file, asar_meta.instrument_file);

    ProductHeader sph = {};
    sph.Load(MPH_SIZE, file_data.data() + MPH_SIZE, SPH_SIZE);

    printf("SPH = \n");
    sph.PrintValues();


    asar_meta.start_nadir_lat = NadirLLParse(sph.Get("START_LAT"));
    asar_meta.start_nadir_lon = NadirLLParse(sph.Get("START_LONG"));
    asar_meta.stop_nadir_lat = NadirLLParse(sph.Get("STOP_LAT"));
    asar_meta.stop_nadir_lon = NadirLLParse(sph.Get("STOP_LONG"));

    asar_meta.ascending = asar_meta.start_nadir_lat < asar_meta.stop_nadir_lat;


    asar_meta.product_name = mph.Get("PRODUCT");
    asar_meta.swath = sph.Get("SWATH");

    int swath_idx = 1;
    if(asar_meta.swath != "IS2")
    {
        printf("Swath mode change support TODO!\n");
        exit(1);
    }

    asar_meta.acquistion_station = mph.Get("ACQUISITION_STATION");
    asar_meta.processing_station = mph.Get("PROC_CENTER");
    asar_meta.polarization = sph.Get("TX_RX_POLAR");



    auto dsds = extract_dsds(sph);

    auto mdsr = dsds.at(0);
    size_t offset = mdsr.ds_offset;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(file_data.data()) + offset;
    const uint8_t* it = reinterpret_cast<const uint8_t*>(file_data.data()) + offset;

    double i_sum = 0.0;
    double q_sum = 0.0;
    double i_sq = 0.0;
    double q_sq = 0.0;
    size_t n_tot = 0;

    std::vector<EchoMeta> echos;
    for(size_t i = 0; mdsr.num_dsr; i++)
    {

        EchoMeta echo_meta = {};
        // Source: ENVISAT-1 ASAR INTERPRETATION OF SOURCE PACKET DATA
        // PO-TN-MMS-SR-0248
        it = CopyBSwapPOD(echo_meta.isp_sensing_time, it);

        FEPAnnotations fep;
        it = CopyBSwapPOD(fep, it);

        uint16_t packet_id;
        it = CopyBSwapPOD(packet_id, it);

        uint16_t sequence_control;
        it = CopyBSwapPOD(sequence_control, it);

        uint16_t packet_length;
        it = CopyBSwapPOD(packet_length, it);

        uint16_t datafield_length;
        it = CopyBSwapPOD(datafield_length, it);

        if(datafield_length != 29)
        {
            break;
        }

        it++;
        echo_meta.mode_id = *it;
        it++;



        if(echo_meta.mode_id != 0x54)
        {
            printf("mode id = %02X - not IM mode!\n", echo_meta.mode_id);
            exit(1);
        }
        echo_meta.onboard_time = 0;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[0]) << 32;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[1]) << 24;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[2]) << 16;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[3]) << 8;
        echo_meta.onboard_time |= static_cast<uint64_t>(it[4]) << 0;

        it += 5;
        it++; // spare byte after 40 bit onboard time

        echo_meta.mode_count = 0;
        echo_meta.mode_count |= static_cast<uint32_t>(it[0]) << 16;
        echo_meta.mode_count |= static_cast<uint32_t>(it[1]) << 8;
        echo_meta.mode_count |= static_cast<uint32_t>(it[2]) << 0;
        it += 3;

        echo_meta.antenna_beam_set_no = *it >> 2;
        echo_meta.comp_ratio = *it & 0x3;
        it++;


        echo_meta.echo_flag = *it & 0x80;
        echo_meta.noise_flag = *it & 0x40;
        echo_meta.cal_flag =  *it & 0x20;
        echo_meta.cal_type = *it & 0x10;
        echo_meta.cycle_count = 0;
        echo_meta.cycle_count |= (it[0] & 0xF) << 8;
        echo_meta.cycle_count |= it[1] << 0;
        it += 2;

        it = CopyBSwapPOD(echo_meta.pri_code, it);
        it = CopyBSwapPOD(echo_meta.swst_code, it);
        it = CopyBSwapPOD(echo_meta.echo_window_code, it);


        {
            uint16_t data;
            it = CopyBSwapPOD(data, it);
            echo_meta.upconverter_raw = data >> 12;
            echo_meta.downconverter_raw = (data >> 7) & 0x1F;
            echo_meta.tx_pol = data & 0x40;
            echo_meta.rx_pol = data & 0x20;
            echo_meta.cal_row_number = data & 0x1F;
        }
        {
            uint16_t data = -1;
            it = CopyBSwapPOD(data, it);
            echo_meta.tx_pulse_code = data >> 6;
            echo_meta.beam_adj_delta = data & 0x3F;
        }

        echo_meta.chirp_pulse_bw_code = it[0];
        echo_meta.aux_tx_monitor_level = it[1];
        it += 2;

        it = CopyBSwapPOD(echo_meta.resampling_factor, it);

        size_t data_len =  packet_length - 29;

        if(echo_meta.echo_flag) {
            size_t n_blocks = data_len / 64;

            for (size_t i = 0; i < n_blocks; i++) {
                const uint8_t* block_data = it + i * 64;
                uint8_t block_id = block_data[0];
//            printf("blockid = %d\n", block_id);
                for (size_t j = 0; j < 63; j++) {
                    uint8_t i_codeword = block_data[1 + j] >> 4;
                    uint8_t q_codeword = block_data[1 + j] & 0xF;

                    float i_samp = ins_file.fbp.i_LUT_fbaq4[FBAQ4Idx(block_id, i_codeword)];
                    float q_samp = ins_file.fbp.q_LUT_fbaq4[FBAQ4Idx(block_id, q_codeword)];
                    //float i_samp = ins_file.fbp.no_adc_fbaq4[FBAQ4Idx(block_id, i_codeword)];
                    //float q_samp = ins_file.fbp.no_adc_fbaq4[FBAQ4Idx(block_id, q_codeword)];
                    echo_meta.raw_data.push_back({i_samp, q_samp});
                    i_sum += i_samp;
                    q_sum += q_samp;

                    i_sq += i_samp * i_samp;
                    q_sq += q_samp * q_samp;
                    n_tot++;
                    //printf("%01X %01X\n", i_sample, q_sample);
                }

            }


            echos.push_back(std::move(echo_meta));
        }

        it += (fep.isp_length - 29);

        //echos.push_back(std::move(echo_meta));

    }


    // i_bias = 3.33e-4
    // q_bias = 5.78e-4
    printf("should be = %f %f\n", 3.33e-4, 5.78e-4);
    printf("SUM I Q, bias !? = %f %f\n", i_sum / n_tot, q_sum / n_tot);


    printf("gain = %f %f\n", i_sq/q_sq, sqrt(i_sq/q_sq));
    //exit(1);

    int idx = 0;
    size_t prev = -1;


    int x = echos.front().raw_data.size();
    int y = echos.size();
    uint16_t min_swst = UINT16_MAX;
    uint16_t max_swst = 0;
    size_t max_samples = 0;
    uint16_t swst_changes = 0;
    uint16_t prev_swst = echos.front().swst_code;

    asar_meta.first_swst_code = echos.front().swst_code;
    asar_meta.last_swst_code = echos.back().swst_code;
    asar_meta.tx_bw_code = echos.front().chirp_pulse_bw_code;
    asar_meta.up_code = echos.front().upconverter_raw;
    asar_meta.down_code = echos.front().downconverter_raw;
    asar_meta.pri_code = echos.front().pri_code;
    asar_meta.tx_pulse_len_code = echos.front().tx_pulse_code;
    asar_meta.beam_set_num_code = echos.front().antenna_beam_set_no;
    asar_meta.beam_adj_code = echos.front().beam_adj_delta;
    asar_meta.resamp_code = echos.front().resampling_factor;
    asar_meta.first_mjd = echos.front().isp_sensing_time;
    asar_meta.first_sbt = echos.front().onboard_time;
    asar_meta.last_mjd = echos.back().isp_sensing_time;
    asar_meta.last_sbt = echos.back().onboard_time;

    std::vector<float> v;
    for(auto& e : echos)
    {
        if(prev_swst != e.swst_code)
        {
            prev_swst = e.swst_code;
            swst_changes++;
        }
        min_swst = std::min(min_swst, e.swst_code);
        max_swst = std::max(max_swst, e.swst_code);
        max_samples = std::max(max_samples, e.raw_data.size());
    }

    asar_meta.swst_changes = swst_changes;

    double pulse_bw = 16e6 / 255  * echos.front().chirp_pulse_bw_code;


    sar_meta.carrier_frequency = ins_file.flp.radar_frequency;
    sar_meta.chirp.range_sampling_rate = ins_file.flp.radar_sampling_rate;

    sar_meta.chirp.pulse_bandwidth = pulse_bw;
    sar_meta.pulse_repetition_frequency = sar_meta.chirp.range_sampling_rate / echos.front().pri_code;


    sar_meta.chirp.coefficient[1] = ins_file.flp.im_chirp[swath_idx].phase[2] * 2;
    sar_meta.chirp.pulse_duration = ins_file.flp.im_chirp[swath_idx].duration;
    sar_meta.chirp.n_samples = sar_meta.chirp.pulse_duration / (1 / sar_meta.chirp.range_sampling_rate);


    double calc_bw = sar_meta.chirp.coefficient[1] * sar_meta.chirp.pulse_duration;
    printf("calc bw = %f , meta bw = %f\n", calc_bw, pulse_bw);

    size_t range_sz = max_samples + max_swst - min_swst;
    printf("range sz = %zu\n", range_sz);



    constexpr double c = 299792458;

    constexpr uint32_t im_idx = 0;
    const uint32_t n_pulses_swst = ins_file.fbp.mode_timelines[im_idx].r_values[swath_idx]; // TODO INS file

    asar_meta.swst_rank = n_pulses_swst;

    //constexpr double rgd = 3.538883113418309e-07;
    asar_meta.two_way_slant_range_time = (min_swst + n_pulses_swst * echos.front().pri_code) * (1 / sar_meta.chirp.range_sampling_rate);

    sar_meta.slant_range_first_sample = asar_meta.two_way_slant_range_time * c / 2;
    sar_meta.wavelength = c / sar_meta.carrier_frequency;
    sar_meta.range_spacing = c / ( 2 * sar_meta.chirp.range_sampling_rate);

    sar_meta.azimuth_spacing = sar_meta.platform_velocity * 0.88 * (1/sar_meta.pulse_repetition_frequency);

    sar_meta.img.range_size = range_sz;
    sar_meta.img.azimuth_size = echos.size();

    img_data.resize(sar_meta.img.range_size * sar_meta.img.azimuth_size);

    for(size_t y = 0; y < echos.size(); y++)
    {
        const auto& e = echos[y];
        size_t idx = y * range_sz;
        idx += e.swst_code - min_swst;

        memcpy(&img_data[idx], e.raw_data.data(), e.raw_data.size() * 8);
    }

    std::cout << "SENSING START = " <<  asar_meta.sensing_start << "\n";
    std::cout << "First echo = " << MjdToPtime(echos.front().isp_sensing_time) << "\n";

    for(size_t i = 0; i < 20; i ++ )
    {
        std::cout << MjdToPtime(echos[i].isp_sensing_time) << "\n";
    }

    //TODO init guess handling?
    double init_guess_lat = asar_meta.start_nadir_lat;
    double init_guess_lon = asar_meta.start_nadir_lon;
    if(asar_meta.ascending)
    {
        init_guess_lat += 3.0;
        init_guess_lon += 1.5;
    }
    else
    {
        init_guess_lat -= 3.0;
        init_guess_lon -= 1.5;
    }

    auto init_xyz = Geo2xyzWgs84(init_guess_lat, init_guess_lon, 0);
    double center_s = (1 / sar_meta.pulse_repetition_frequency) * sar_meta.img.azimuth_size / 2;
    auto center_time = asar_meta.sensing_start + boost::posix_time::microseconds(static_cast<uint32_t>(center_s * 1e6));


    //TODO investigate fast time effect on geolocation
    double slant_range_center = sar_meta.slant_range_first_sample + ((sar_meta.img.range_size - sar_meta.chirp.n_samples) / 2 - 0) * sar_meta.range_spacing;

    auto osv = InterpolateOrbit(sar_meta.osv, center_time);

    sar_meta.center_point = RangeDopplerGeoLocate({osv.x_vel, osv.y_vel, osv.z_vel}, {osv.x_pos, osv.y_pos, osv.z_pos}, init_xyz, slant_range_center);
    sar_meta.center_time = center_time;
    sar_meta.first_line_time = asar_meta.sensing_start;
    sar_meta.azimuth_bandwidth_fraction = 0.8f;
    auto llh = xyz2geoWGS84(sar_meta.center_point);
    printf("center point = %f %f\n", llh.latitude, llh.longitude);
    printf("tx pulse calc dur = %f\n", asar_meta.tx_pulse_len_code / (1/sar_meta.chirp.range_sampling_rate));
}
