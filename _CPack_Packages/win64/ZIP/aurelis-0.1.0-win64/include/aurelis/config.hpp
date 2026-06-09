#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/errors.hpp"
#include "aurelis/nlohmann/json.hpp"
#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>

namespace aurelis {

using json = nlohmann::json;

namespace detail {

template <typename T>
inline void require_positive(T value, const std::string& field_name) {
    if constexpr (std::is_integral_v<T>) {
        if (value <= static_cast<T>(0)) {
            throw AurelisException(
                ErrorCode::InvalidConfig,
                "Invalid config value for '" + field_name + "': expected a positive integer");
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        if (!(value > static_cast<T>(0))) {
            throw AurelisException(
                ErrorCode::InvalidConfig,
                "Invalid config value for '" + field_name + "': expected a positive number");
        }
    } else {
        static_assert(std::is_arithmetic_v<T>, "require_positive expects an arithmetic type");
    }
}

}  // namespace detail

struct AurelisConfig {
    aurelis::lens::LensConfig lens;

    bool save(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs) {
            return false;
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
        return ofs.good();
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

                detail::require_positive(cfg.lens.vocab_size, "lens.vocab_size");
                detail::require_positive(cfg.lens.D, "lens.D");
                detail::require_positive(cfg.lens.d_model, "lens.d_model");
                detail::require_positive(cfg.lens.d_tau, "lens.d_tau");
                detail::require_positive(cfg.lens.d_ff, "lens.d_ff");
                detail::require_positive(cfg.lens.num_layers, "lens.num_layers");
                detail::require_positive(cfg.lens.num_scales, "lens.num_scales");
                detail::require_positive(cfg.lens.lambda_stab, "lens.lambda_stab");
                detail::require_positive(cfg.lens.lambda_aux, "lens.lambda_aux");
                detail::require_positive(cfg.lens.lr, "lens.lr");
            }
            
            return cfg;
        } catch (const nlohmann::parse_error& e) {
            throw AurelisException(ErrorCode::InvalidJson, "Failed to parse config JSON: " + std::string(e.what()));
        } catch (const nlohmann::type_error& e) {
            throw AurelisException(ErrorCode::InvalidConfig, "Invalid config format: " + std::string(e.what()));
        }
    }
};

} // namespace aurelis
