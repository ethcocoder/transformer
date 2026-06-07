#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/tensor.hpp"
#include "aurelis/errors.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

namespace aurelis {

struct AurelisConfig {
    LensConfig lens;

    void save(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs) {
            throw AurelisException(ErrorCode::FileNotFound, "Could not open config file for writing: " + path);
        }
        ofs << "{\n";
        ofs << "  \"lens\": {\n";
        ofs << "    \"vocab_size\": " << lens.vocab_size << ",\n";
        ofs << "    \"D\": " << lens.D << ",\n";
        ofs << "    \"d_model\": " << lens.d_model << ",\n";
        ofs << "    \"d_tau\": " << lens.d_tau << ",\n";
        ofs << "    \"d_ff\": " << lens.d_ff << ",\n";
        ofs << "    \"num_layers\": " << lens.num_layers << ",\n";
        ofs << "    \"num_scales\": " << lens.num_scales << ",\n";
        ofs << "    \"lambda_stab\": " << lens.lambda_stab << ",\n";
        ofs << "    \"lambda_aux\": " << lens.lambda_aux << ",\n";
        ofs << "    \"lr\": " << lens.lr << "\n";
        ofs << "  }\n";
        ofs << "}\n";
    }

    static AurelisConfig load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) {
            throw AurelisException(ErrorCode::FileNotFound, "Could not open config file for reading: " + path);
        }

        std::map<std::string, std::string> key_values;
        std::string line;
        while (std::getline(ifs, line)) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                std::string value = line.substr(colon_pos + 1);
                
                // Trim whitespace and quotes
                auto trim = [](std::string s) {
                    s.erase(0, s.find_first_not_of(" \t\n\r\"{,}"));
                    s.erase(s.find_last_not_of(" \t\n\r\"{,}") + 1);
                    return s;
                };

                key = trim(key);
                value = trim(value);
                
                if (!key.empty() && !value.empty()) {
                    key_values[key] = value;
                }
            }
        }

        AurelisConfig cfg;

        auto get_int = [&](const std::string& key, int default_val) -> int {
            auto it = key_values.find(key);
            if (it != key_values.end()) {
                try {
                    return std::stoi(it->second);
                } catch (...) {}
            }
            return default_val;
        };

        auto get_float = [&](const std::string& key, float default_val) -> float {
            auto it = key_values.find(key);
            if (it != key_values.end()) {
                try {
                    return std::stof(it->second);
                } catch (...) {}
            }
            return default_val;
        };

        cfg.lens.vocab_size = get_int("vocab_size", 16);
        cfg.lens.D = get_int("D", 64);
        cfg.lens.d_model = get_int("d_model", 32);
        cfg.lens.d_tau = get_int("d_tau", 32);
        cfg.lens.d_ff = get_int("d_ff", 256);
        cfg.lens.num_layers = get_int("num_layers", 2);
        cfg.lens.num_scales = get_int("num_scales", 4);
        cfg.lens.lambda_stab = get_float("lambda_stab", 0.01f);
        cfg.lens.lambda_aux = get_float("lambda_aux", 0.001f);
        cfg.lens.lr = get_float("lr", 0.01f);

        return cfg;
    }
};

} // namespace aurelis
