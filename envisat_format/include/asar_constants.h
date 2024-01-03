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

#include <cstdint>
#include <string_view>

/**
 * ASAR specific constants use for converting raw data to physical values
 */

// PO-TN-MMS-SR-0248 - conversion tables
float UpconverterLevelConversion(uint8_t up);

float AuxTxMonitorConversion(uint16_t aux_tx);

namespace alus::asar::specification {

enum ProductTypes { SAR_IM0, ASA_IM0, SAR_IMS, SAR_IMP, ASA_IMS, ASA_IMP, UNIDENTIFIED };

enum Instrument { SAR, ASAR };

constexpr std::string_view PRODUCT_NAME_SAR_IM0{"SAR_IM__0"};
constexpr std::string_view PRODUCT_NAME_ASA_IM0{"ASA_IM__0"};

ProductTypes GetProductTypeFrom(std::string_view product_name);
ProductTypes TryDetermineTargetProductFrom(ProductTypes in_product, std::string_view user_defined_target_type);
Instrument GetInstrumentFrom(ProductTypes product);
}  // namespace alus::asar::specification
