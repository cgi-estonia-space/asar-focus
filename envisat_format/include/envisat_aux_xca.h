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

#include "bswap_util.h"
#include "envisat_types.h"

namespace alus::asar::envformat::aux {

    // PO-RS-MDA-GS-2009 table 8.6.2.1-1
    struct ExternalCalibration {
        mjd creation_time;
        ul dsr_length;
        // 7 values stand for swaths IS1 to IS7.
        fl scaling_factor_im_slc_hh[7];
        fl scaling_factor_im_slc_vv[7];
        fl scaling_factor_im_precision_hh[7];
        fl scaling_factor_im_precision_vv[7];
        fl scaling_factor_im_geocoded_hh[7];
        fl scaling_factor_im_geocoded_vv[7];
        fl scaling_factor_im_mediumres_hh[7];
        fl scaling_factor_im_mediumres_vv[7];
        fl scaling_factor_ap_slc_hh[7];
        fl scaling_factor_ap_slc_vv[7];
        fl scaling_factor_ap_slc_hv[7];
        fl scaling_factor_ap_slc_vh[7];
        fl scaling_factor_ap_precision_hh[7];
        fl scaling_factor_ap_precision_vv[7];
        fl scaling_factor_ap_precision_hv[7];
        fl scaling_factor_ap_precision_vh[7];
        fl scaling_factor_ap_geocoded_hh[7];
        fl scaling_factor_ap_geocoded_vv[7];
        fl scaling_factor_ap_geocoded_hv[7];
        fl scaling_factor_ap_geocoded_vh[7];
        fl scaling_factor_ap_mediumres_hh[7];
        fl scaling_factor_ap_mediumres_vv[7];
        fl scaling_factor_ap_mediumres_hv[7];
        fl scaling_factor_ap_mediumres_vh[7];
        fl scaling_factor_wv_hh[7];
        fl scaling_factor_wv_vv[7];
        fl scaling_factor_ws_hh;
        fl scaling_factor_ws_vv;
        fl scaling_factor_gm_hh;
        fl scaling_factor_gm_vv;

        fl ref_elev_angle_is1;
        fl ref_elev_angle_is2;
        fl ref_elev_angle_is3_ss2;
        fl ref_elev_angle_is4_ss3;
        fl ref_elev_angle_is5_ss4;
        fl ref_elev_angle_is6_ss5;
        fl ref_elev_angle_is7;
        fl ref_elev_angle_ss1;

        // pattern is defined from reference elevation angle - 5 deg. to reference
        // elevation angle + 5 deg. in 0.05 degree steps.
        fl two_way_ant_elev_pat_gain_lut_is1[804];
        fl two_way_ant_elev_pat_gain_lut_is2[804];
        fl two_way_ant_elev_pat_gain_lut_is3_ss2[804];
        fl two_way_ant_elev_pat_gain_lut_is4_ss3[804];
        fl two_way_ant_elev_pat_gain_lut_is5_ss4[804];
        fl two_way_ant_elev_pat_gain_lut_is6_ss5[804];
        fl two_way_ant_elev_pat_gain_lut_is7[804];
        fl two_way_ant_elev_pat_gain_lut_ss1[804];
        uc spare[32];

        void Bswap() {
            creation_time = bswap(creation_time);
            dsr_length = bswap(dsr_length);
            BSWAP_ARR(scaling_factor_im_slc_hh);
            BSWAP_ARR(scaling_factor_im_slc_vv);
            BSWAP_ARR(scaling_factor_im_precision_hh);
            BSWAP_ARR(scaling_factor_im_precision_vv);
            BSWAP_ARR(scaling_factor_im_geocoded_hh);
            BSWAP_ARR(scaling_factor_im_geocoded_vv);
            BSWAP_ARR(scaling_factor_im_mediumres_hh);
            BSWAP_ARR(scaling_factor_im_mediumres_vv);
            BSWAP_ARR(scaling_factor_ap_slc_hh);
            BSWAP_ARR(scaling_factor_ap_slc_vv);
            BSWAP_ARR(scaling_factor_ap_slc_hv);
            BSWAP_ARR(scaling_factor_ap_slc_vh);
            BSWAP_ARR(scaling_factor_ap_precision_hh);
            BSWAP_ARR(scaling_factor_ap_precision_vv);
            BSWAP_ARR(scaling_factor_ap_precision_hv);
            BSWAP_ARR(scaling_factor_ap_precision_vh);
            BSWAP_ARR(scaling_factor_ap_geocoded_hh);
            BSWAP_ARR(scaling_factor_ap_geocoded_vv);
            BSWAP_ARR(scaling_factor_ap_geocoded_hv);
            BSWAP_ARR(scaling_factor_ap_geocoded_vh);
            BSWAP_ARR(scaling_factor_ap_mediumres_hh);
            BSWAP_ARR(scaling_factor_ap_mediumres_vv);
            BSWAP_ARR(scaling_factor_ap_mediumres_hv);
            BSWAP_ARR(scaling_factor_ap_mediumres_vh);
            BSWAP_ARR(scaling_factor_wv_hh);
            BSWAP_ARR(scaling_factor_wv_vv);
            scaling_factor_ws_hh = bswap(scaling_factor_ws_hh);
            scaling_factor_ws_vv = bswap(scaling_factor_ws_vv);
            scaling_factor_gm_hh = bswap(scaling_factor_gm_hh);
            scaling_factor_gm_vv = bswap(scaling_factor_gm_vv);

            ref_elev_angle_is1 = bswap(ref_elev_angle_is1);
            ref_elev_angle_is2 = bswap(ref_elev_angle_is2);
            ref_elev_angle_is3_ss2 = bswap(ref_elev_angle_is3_ss2);
            ref_elev_angle_is4_ss3 = bswap(ref_elev_angle_is4_ss3);
            ref_elev_angle_is5_ss4 = bswap(ref_elev_angle_is5_ss4);
            ref_elev_angle_is6_ss5 = bswap(ref_elev_angle_is6_ss5);
            ref_elev_angle_is7 = bswap(ref_elev_angle_is7);
            ref_elev_angle_ss1 = bswap(ref_elev_angle_ss1);

            // pattern is defined from reference elevation angle - 5 deg. to reference
            // elevation angle + 5 deg. in 0.05 degree steps.
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is1);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is2);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is3_ss2);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is4_ss3);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is5_ss4);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is6_ss5);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_is7);
            BSWAP_ARR(two_way_ant_elev_pat_gain_lut_ss1);
        }
    };

    static_assert(sizeof(ExternalCalibration) == 26552);
}
