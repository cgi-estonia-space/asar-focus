/**
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c) by CGI Estonia AS
 *
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity is licensed under a
 * Creative Commons Attribution-ShareAlike 4.0 International License.
 *
 * You should have received a copy of the license along with this
 * work. If not, see http://creativecommons.org/licenses/by-sa/4.0/
 */

#include "asar_constants.h"

#include <array>

#include <boost/algorithm/string/predicate.hpp>

// PO-TN-MMS-SR-0248 - conversion tables

float UpconverterLevelConversion(uint8_t up) {
    constexpr std::array<float, 16> lut = {
        25.1625, 25.6825, 26.2025, 26.7225, 27.2425, 27.7625, 28.2825, 28.8025,
        29.3225, 29.8425, 30.3625, 30.8825, 31.4025, 31.9225, 32.4425, 32.9625,
    };
    return lut.at(up);
}

float AuxTxMonitorConversion(uint16_t aux_tx) {
    std::array<float, 260> lut =

        {
            -19.036, -19.011, -18.987, -18.962, -18.938, -18.914, -18.889, -18.865, -18.841, -18.817, -18.793, -18.769,
            -18.745, -18.722, -18.698, -18.674, -18.651, -18.627, -18.604, -18.580, -18.557, -18.534, -18.511, -18.488,
            -18.464, -18.441, -18.419, -18.396, -18.373, -18.350, -18.327, -18.305, -18.282, -18.260, -18.237, -18.215,
            -18.192, -18.170, -18.148, -18.126, -18.103, -18.081, -18.059, -18.037, -18.015, -17.994, -17.972, -17.950,
            -17.928, -17.907, -17.885, -17.864, -17.842, -17.821, -17.799, -17.778, -17.757, -17.736, -17.714, -17.693,
            -17.672, -17.651, -17.630, -17.609, -17.589, -17.568, -17.547, -17.526, -17.506, -17.485, -17.464, -17.444,
            -17.423, -17.403, -17.383, -17.362, -17.342, -17.322, -17.302, -17.282, -17.261, -17.241, -17.221, -17.201,
            -17.182, -17.162, -17.142, -17.122, -17.102, -17.083, -17.063, -17.044, -17.024, -17.005, -16.985, -16.966,
            -16.946, -16.927, -16.908, -16.888, -16.869, -16.850, -16.831, -16.812, -16.793, -16.774, -16.755, -16.736,
            -16.717, -16.698, -16.680, -16.661, -16.642, -16.623, -16.605, -16.586, -16.568, -16.549, -16.531, -16.512,
            -16.494, -16.476, -16.457, -16.439, -16.421, -16.403, -16.384, -16.366, -16.348, -16.330, -16.312, -16.294,
            -16.276, -16.258, -16.241, -16.223, -16.205, -16.187, -16.170, -16.152, -16.134, -16.117, -16.099, -16.082,
            -16.064, -16.047, -16.029, -16.012, -15.994, -15.977, -15.960, -15.943, -15.925, -15.908, -15.891, -15.874,
            -15.857, -15.840, -15.823, -15.806, -15.789, -15.772, -15.755, -15.738, -15.721, -15.705, -15.688, -15.671,
            -15.654, -15.638, -15.621, -15.604, -15.588, -15.571, -15.555, -15.538, -15.522, -15.506, -15.489, -15.473,
            -15.457, -15.440, -15.424, -15.408, -15.392, -15.375, -15.359, -15.343, -15.327, -15.311, -15.295, -15.279,
            -15.263, -15.247, -15.231, -15.215, -15.200, -15.184, -15.168, -15.152, -15.137, -15.121, -15.105, -15.090,
            -15.074, -15.058, -15.043, -15.027, -15.012, -14.996, -14.981, -14.965, -14.950, -14.935, -14.919, -14.904,
            -14.889, -14.874, -14.858, -14.843, -14.828, -14.813, -14.798, -14.783, -14.768, -14.752, -14.737, -14.722,
            -14.708, -14.693, -14.678, -14.663, -14.648, -14.633, -14.618, -14.603, -14.589, -14.574, -14.559, -14.545,
            -14.530, -14.515, -14.501, -14.486, -14.472, -14.457, -14.442, -14.428, -14.414, -14.399, -14.385, -14.370,
            -14.356, -14.342, -14.327, -14.313, -14.299, -14.284, -14.270, -14.256,
        };

    return lut.at(aux_tx);
}

namespace alus::asar::specification {

ProductTypes GetProductTypeFrom(std::string_view product_name) {
    if (boost::algorithm::starts_with(product_name, PRODUCT_NAME_ASA_IM0)) {
        return ProductTypes::ASA_IM0;
    } else if (boost::algorithm::starts_with(product_name, PRODUCT_NAME_SAR_IM0)) {
        return ProductTypes::SAR_IM0;
    } else {
        return ProductTypes::UNIDENTIFIED;
    }
}

ProductTypes TryDetermineTargetProductFrom(ProductTypes in_product, std::string_view user_defined_target_type) {
    if (in_product == ProductTypes::SAR_IM0) {
        if (boost::iequals(user_defined_target_type, "IMS")) {
            return ProductTypes::SAR_IMS;
        }
    }

    if (in_product == ProductTypes::ASA_IM0) {
        if (boost::iequals(user_defined_target_type, "IMS")) {
            return ProductTypes::ASA_IMS;
        }
    }

    throw std::runtime_error("Focussing '" + std::string(user_defined_target_type) +
                             "' product from given input is not supported or valid.");
}

Instrument GetInstrumentFrom(ProductTypes product) {
    if (product == ProductTypes::SAR_IM0 || product == ProductTypes::SAR_IMS) {
        return Instrument::SAR;
    }

    if (product == ProductTypes::ASA_IM0 || product == ProductTypes::ASA_IMS) {
        return Instrument::ASAR;
    }

    throw std::runtime_error("Unsupported product type supplied to " + std::string(__FUNCTION__));
}

}  // namespace alus::asar::specification