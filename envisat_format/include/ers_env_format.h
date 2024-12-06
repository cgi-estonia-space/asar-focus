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

#include <cstddef>

namespace alus::asar::envformat::ers {

namespace highrate {
// PX-SP-50-9105 3/1
// https://earth.esa.int/eogateway/documents/20142/37627/ERS-products-specification-with-Envisat-format.pdf
constexpr size_t MDSR_SIZE_BYTES{11498};
constexpr size_t PROCESSOR_ANNOTATION_ISP_SIZE_BYTES{12};
constexpr size_t FEP_ANNOTATION_SIZE_BYTES{20};
constexpr size_t MDSR_ISP_SIZE_BYTES{11466};  // Adding ISP sensing (12) and FEP annotations (20) one gets sum above.
constexpr size_t DATA_RECORD_NUMBER_SIZE_BYTES{4};
constexpr size_t IDHT_HEADER_SIZE_BYTES{10};
constexpr size_t SBT_OFFSET_BYTES{PROCESSOR_ANNOTATION_ISP_SIZE_BYTES + FEP_ANNOTATION_SIZE_BYTES + 16};
constexpr size_t AUXILIARY_REPLICA_CALIBRATION_SIZE_BYTES{220};
constexpr size_t MEASUREMENT_DATA_SIZE_BYTES{11232};
constexpr size_t MEASUREMENT_DATA_SAMPLE_COUNT{5616};
static_assert(DATA_RECORD_NUMBER_SIZE_BYTES + IDHT_HEADER_SIZE_BYTES + AUXILIARY_REPLICA_CALIBRATION_SIZE_BYTES +
                  MEASUREMENT_DATA_SIZE_BYTES ==
              MDSR_ISP_SIZE_BYTES);
static_assert(MEASUREMENT_DATA_SAMPLE_COUNT * 2 == MEASUREMENT_DATA_SIZE_BYTES);

}  // namespace highrate

}  // namespace alus::asar::envformat::ers