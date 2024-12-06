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

#include <boost/algorithm/string.hpp>
#include <cstdlib>
#include <string>

#include "checks.h"

constexpr size_t MPH_SIZE = 1247;
constexpr size_t SPH_SIZE = 1956;

constexpr size_t DSD_SIZE = 280;

struct DSD_lvl0 {
    std::string ds_name;
    std::string ds_type;
    std::string filename;
    size_t ds_offset;
    size_t ds_size;
    size_t num_dsr;
    size_t dsr_size;
};

class ProductHeader {
public:
    void Load(const char* data, uint32_t n) {
        copy_ = std::string(data, n);
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, copy_, boost::is_any_of("\n"));

        for (const auto& e : tokens) {
            auto idx = e.find('=');
            if (idx == std::string::npos) {
                continue;
            }

            auto value = e.substr(idx + 1, std::string::npos);
            if (value.back() == '"') {
                value.pop_back();
            }
            if (value.front() == '"') {
                value.erase(value.begin());
            }
            kv_vec_.push_back({e.substr(0, idx), value});
        }
    }

    std::string Get(std::string key) {
        for (const auto& kv : kv_vec_) {
            if (kv.key == key) {
                return kv.value;
            }
        }
        ERROR_EXIT(key + " not found in Lvl1MPH/SPH file");
    }

    const auto& GetIdx(size_t idx) { return kv_vec_.at(idx); }

    const char* Data() const { return copy_.c_str(); }

private:
    struct KV {
        std::string key;
        std::string value;
    };

    std::vector<KV> kv_vec_;
    std::string copy_;
};

inline size_t extract_bytes(const std::string& in) {
    auto idx = in.find('=');
    auto idx_end = in.find("<bytes>");
    for (size_t i = idx; i < in.size(); i++) {
        if (in[i] >= '1' && in[i] <= '9') {
            return std::stoul(in.substr(i, idx_end - i));
        }
    }
    return 0;
}

inline size_t extract_num(const std::string& in) {
    auto idx = in.find('=');
    auto idx_end = in.find("<bytes>");
    (void)idx_end;
    for (size_t i = idx; i < in.size(); i++) {
        if (in[i] >= '1' && in[i] <= '9') {
            return std::stoul(in.substr(i, in.size()));
        }
    }
    return 0;
}

inline std::vector<DSD_lvl0> ExtractDSDs(const ProductHeader& sph) {
    size_t dsd_offset = SPH_SIZE - DSD_SIZE * 4;

    std::vector<DSD_lvl0> res;

    for (size_t i = 0; i < 4; i++) {
        std::vector<std::string> tokens;
        std::string in(sph.Data() + dsd_offset + i * DSD_SIZE, DSD_SIZE);
        boost::algorithm::split(tokens, in, boost::is_any_of("\n"));
        if (tokens.size() < 6) {
            continue;
        }
        DSD_lvl0 dsd = {};
        dsd.ds_name = tokens.at(0);
        dsd.ds_type = tokens.at(1);
        dsd.filename = tokens.at(2);
        dsd.ds_offset = extract_bytes(tokens.at(3));
        dsd.ds_size = extract_bytes(tokens.at(4));
        dsd.num_dsr = extract_num(tokens.at(5));
        dsd.dsr_size = extract_bytes(tokens.at(6));

        res.push_back(dsd);
    }

    return res;
}