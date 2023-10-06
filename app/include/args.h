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

#include <boost/program_options.hpp>

namespace alus::asar {

    class Args final {
    public:
        Args() = delete;

        [[nodiscard]] std::string_view GetInputDsPath() const { return ds_path_; }
        [[nodiscard]] std::string_view GetAuxPath() const { return aux_path_; }
        [[nodiscard]] std::string_view GetOrbitPath() const { return orbit_path_; }
        [[nodiscard]] std::string_view GetOutputPath() const { return output_path_; }
        [[nodiscard]] std::optional<std::string_view> GetProcessSensingStart() const { process_sensing_start_.empty() ? return std::nullopt : return process_sensing_start_; }

    private:

        boost::program_options::options_description args_{""};

        std::string ds_path_{};
        std::string aux_path_{};
        std::string orbit_path_{};
        std::string output_path_{};
        std::string process_sensing_start_{};
        std::string process_sensing_end_{};
    };

}