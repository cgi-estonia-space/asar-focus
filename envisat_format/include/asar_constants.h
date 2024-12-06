/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

constexpr double REFERENCE_RANGE = 800000;

inline bool IsSLCProduct(ProductTypes product)
{
    return product == SAR_IMS || product == ASA_IMS;
}

inline bool IsDetectedProduct(ProductTypes product)
{
    return product == SAR_IMP || product == ASA_IMP;
}

constexpr std::string_view PRODUCT_NAME_SAR_IM0{"SAR_IM__0"};
constexpr std::string_view PRODUCT_NAME_ASA_IM0{"ASA_IM__0"};

ProductTypes GetProductTypeFrom(std::string_view product_name);
ProductTypes TryDetermineTargetProductFrom(ProductTypes in_product, std::string_view user_defined_target_type);
Instrument GetInstrumentFrom(ProductTypes product);
}  // namespace alus::asar::specification
