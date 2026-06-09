#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/errors.hpp"
#include "aurelis/nlohmann/json.hpp"
#include <fstream>
#include <string>

namespace aurelis {

using json = nlohmann::json;

struct AurelisConfig {
    LensConfig lens;

    void save(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs) {
            throw AurelisException(ErrorCode::FileNotFound, "Could not open config file for writing: " + path);
        }
        
        json j;
        j["lens"]["vocab_size"] = lens.vocab_size;
        j["lens"]["D"] = lens.D;
        j["lens"]["d_model"] = lens.d_model;
        j["lens"]["d_tau"] = lens.d_tau;
        j["lens"]["d_ff"] = lens.d_ff;
        j["lens"]["num_layers"] = lens.num_layers;
        j["lens"]["num_scales"] = lens.num_scales;
        j["lens"]["lambda_stab"] = lens.lambda_stab;
        j["lens"]["lambda_aux"] = lens.lambda_aux;
        j["lens"]["lr"] = lens.lr;
        
        ofs << j.dump(4) << std::endl;
    }

    static AurelisConfig load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) {
            throw AurelisException(ErrorCode::FileNotFound, "Could not open config file for reading: " + path);
        }

        try {
            json j;
            ifs >> j;
            
            AurelisConfig cfg;
            
            if (j.contains("lens")) {
                auto& lens_j = j["lens"];
                cfg.lens.vocab_size = lens_j.value("vocab_size", 16);
                cfg.lens.D = lens_j.value("D", 64);
                cfg.lens.d_model = lens_j.value("d_model", 32);
                cfg.lens.d_tau = lens_j.value("d_tau", 32);
                cfg.lens.d_ff = lens_j.value("d_ff", 256);
                cfg.lens.num_layers = lens_j.value("num_layers", 2);
                cfg.lens.num_scales = lens_j.value("num_scales", 4);
                cfg.lens.lambda_stab = lens_j.value("lambda_stab", 0.01f);
                cfg.lens.lambda_aux = lens_j.value("lambda_aux", 0.001f);
                cfg.lens.lr = lens_j.value("lr", 0.01f);
            }
            
            return cfg;
        } catch (const json::parse_error& e) {
            throw AurelisException(ErrorCode::InvalidJson, "Failed to parse config JSON: " + std::string(e.what()));
        } catch (const json::type_error& e) {
            throw AurelisException(ErrorCode::InvalidConfig, "Invalid config format: " + std::string(e.what()));
        }
    }
};

} // namespace aurelis
