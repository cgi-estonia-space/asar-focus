#pragma once

#include <cstdint>
#include <string_view>


/**
 * ASAR specific constants use for converting raw data to physical values
 */

// PO-TN-MMS-SR-0248 - conversion tables
float UpconverterLevelConversion(uint8_t up);

float AuxTxMonitorConversion(uint16_t aux_tx);

namespace alus::asar::specification {

    enum ProductTypes{
        SAR_IM0,
        ASA_IM0,
        UNIDENTIFIED
    };

    constexpr std::string_view PRODUCT_NAME_SAR_IM0{"SAR_IM__0"};
    constexpr std::string_view PRODUCT_NAME_ASA_IM0{"ASA_IM__0"};

    ProductTypes GetProductTypeFrom(std::string_view product_name);
}
