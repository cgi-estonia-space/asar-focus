//
// Created by priit on 5/24/23.
//

#include "ers_im_lvl1_file.h"

#include "envisat_ph.h"





void WriteLvl1(SARMetadata& meta, MDS& mds)
{
    ERS_IM_LVL1 out = {};

    static_assert(__builtin_offsetof(ERS_IM_LVL1, main_processing_params) == 7516);


    {
        auto& mph = out.mph;
        mph.SetDefaults();
        mph.Set_PRODUCT("SAR_IMS_1PNESA19950215_092916_00000016F142_01959_18760_0000.E1");
        mph.Set_PROC_STAGE('N');
        mph.Set_REF_DOC("test.pdf");


        auto now = boost::posix_time::microsec_clock::universal_time();

        mph.SetDataAcqusitionProcessingInfo("Kapa Kohila", "Antsla", PtimeToStr(now), "TEST 0.0.1" );

        mph.Set_SBT_Defaults();


        boost::posix_time::ptime start(boost::gregorian::date(1995, 1, 1));

        auto stop = start + boost::posix_time::seconds(15);


        mph.SetSensingStartStop(PtimeToStr(start), PtimeToStr(stop));
        mph.Set_ORBIT_Defaults();
        mph.Set_SBT_Defaults();
        mph.Set_LEAP_Defaults();
        mph.Set_PRODUCT_ERR('0');


        // set tot size later

        size_t tot = sizeof(out);
        tot += mds.n_records*mds.record_size;
        mph.SetProductSizeInformation(tot, sizeof(out.sph), 18, 6);

    }
    {
        auto& sph = out.sph;
        sph.SetDefaults();
        sph.SetHeader("Image Mode SLC Image", 0, 1, 1);


        boost::posix_time::ptime start(boost::gregorian::date(1995, 1, 1));

        auto stop = start + boost::posix_time::seconds(15);
        sph.SetLineTimeInfo(PtimeToStr(start), PtimeToStr((stop)));

        // todo geopos...
        sph.SetFirstPosition({58.61, 24.97}, {58.73, 24.08}, {58.83, 23.29});
        sph.SetLastPosition({57.63, 24.49}, {23.61, 57.84}, {58.83, 22.84});


        sph.SetProductInfo1("IS2", "DESCENDING", "COMPLEX", "RAN/DOP");
        sph.SetProductInfo2("V/V", "", "NONE");
        sph.SetProductInfo3(1, 1);
        sph.SetProductInfo4(meta.range_spacing, meta.azimuth_spacing, 1.0/meta.pulse_repetition_frequency);
        meta.img.range_size = (mds.record_size - 17) / 4;
        sph.SetProductInfo5(meta.img.range_size, "SWORD");

        out.main_processing_params.general_summary.num_samples_per_line = meta.img.range_size;


        {
            size_t offset = offsetof(decltype(out), summary_quality);
            constexpr size_t size = sizeof(out.summary_quality);
            static_assert(size == 170);
            sph.dsds[0].SetInternalDSD("MDS1 SQ ADS", 'A', offset, 1, size);
        }

        {
            sph.dsds[1].SetEmptyDSD("MDS2 SQ ADS", 'A');
        }

        {
            size_t offset = offsetof(decltype(out), main_processing_params);
            constexpr size_t size = sizeof(out.main_processing_params);
            static_assert(size == 2009);
            sph.dsds[2].SetInternalDSD("MAIN PROCESSING PARAMS ADS", 'A', offset, 1, size);
        }

        {
            size_t offset = offsetof(decltype(out), main_processing_params);
            constexpr size_t size = sizeof(out.main_processing_params);
            static_assert(size == 2009);
            sph.dsds[2].SetInternalDSD("MAIN PROCESSING PARAMS ADS", 'A', offset, 1, size);
        }

        {
            size_t offset = offsetof(decltype(out), dop_centroid_coeffs);
            constexpr size_t size = sizeof(out.dop_centroid_coeffs);
            static_assert(size == 55);
            sph.dsds[3].SetInternalDSD("DOP CENTROID COEFFS ADS", 'A', offset, 1, size);
        }

        sph.dsds[4].SetEmptyDSD("SR GR ADS", 'A');

        {
            size_t offset = offsetof(decltype(out), chirp_params);
            constexpr size_t size = sizeof(out.chirp_params);
            static_assert(size == 1483);
            sph.dsds[5].SetInternalDSD("CHIRP PARAMS ADS", 'A', offset, 1, size);
        }

        sph.dsds[6].SetEmptyDSD("MDS1 ANTENNA ELEV PATT ADS", 'A');
        sph.dsds[7].SetEmptyDSD("MDS2 ANTENNA ELEV PATT ADS", 'A');

        {
            size_t offset = offsetof(decltype(out), geolocation_grid);
            constexpr size_t size = sizeof(out.geolocation_grid[0]);
            size_t n = std::size(out.geolocation_grid);
            static_assert(size == 521);
            sph.dsds[8].SetInternalDSD("GEOLOCATION GRID ADS", 'A', offset, n, size);
        }

        sph.dsds[9].SetEmptyDSD("MAP PROJECTION GADS", 'G');
        {
            size_t offset = sizeof(ERS_IM_LVL1);
            sph.dsds[10].SetInternalDSD("MDS1", 'M', offset, mds.n_records, mds.record_size);
            // MDS1
        }

        sph.dsds[11].SetEmptyDSD("MDS2", 'M');
        sph.dsds[12].SetReferenceDSD("LEVEL 0 PRODUCT", "SAR_IM__0PWDSI19950215_092915_00000018F142_01959_18760_0000.E1");
        sph.dsds[13].SetReferenceDSD("ASAR PROCESSOR CONFIG", "ER1_CON_AXVXXX20100209_000000_19910717_000000_20000310_000000");
        sph.dsds[14].SetReferenceDSD("INSTRUMENT CHARACTERIZATION", "ER1_INS_AXVXXX20110104_154739_19910717_000000_20000310_000000");
        sph.dsds[15].SetEmptyDSD("EXTERNAL CHARACTERIZATION", 'R');
        sph.dsds[16].SetReferenceDSD("EXTERNAL CALIBRATION", "ER1_XCA_AXVXXX20100209_000000_19910717_000000_19980224_000000");
        sph.dsds[17].SetReferenceDSD("ORBIT STATE VECTOR 1", "DOR_VOR_AXVF-P19950214_210000_19950214_210000_19950216_030000");

    }


    FILE* fp = fopen("/tmp/SAR_IMS_1PNESA19950215_092916_00000016F142_01959_18760_0000.E1", "w");

    fwrite(&out, sizeof(out), 1, fp);
    fwrite(mds.buf, mds.record_size, mds.n_records, fp);
    fclose(fp);

    printf("FILE WRITE DONE!\n");
}