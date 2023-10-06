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

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/program_options.hpp>


namespace alus::asar {

    class Args final {
    public:
        Args() = delete;
        Args(const std::vector<char*>& args);

        [[nodiscard]] bool IsHelpRequested() const { return vm_.at("help").as<bool>(); }
        [[nodiscard]] std::string GetHelp() const;
        [[nodiscard]] std::string_view GetInputDsPath() const { return ds_path_; }
        [[nodiscard]] std::string_view GetAuxPath() const { return aux_path_; }
        [[nodiscard]] std::string_view GetOrbitPath() const { return orbit_path_; }
        [[nodiscard]] std::string_view GetOutputPath() const { return output_path_; }
        [[nodiscard]] std::string_view GetFocussedProductType() const { return focussed_product_type_; }
        [[nodiscard]] std::optional<std::string_view> GetProcessSensingStart() const;
        [[nodiscard]] std::optional<std::string_view> GetProcessSensingEnd() const;

    private:

        void Construct();

        boost::program_options::variables_map vm_;
        boost::program_options::options_description args_{""};

        std::string ds_path_{};
        std::string aux_path_{};
        std::string orbit_path_{};
        std::string output_path_{};
        std::string focussed_product_type_{};
        std::string process_sensing_start_{};
        std::string process_sensing_end_{};
    };

}