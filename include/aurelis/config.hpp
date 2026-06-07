#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/tensor.hpp"
#include "aurelis/nlohmann/json.hpp"
#include <fstream>
#include <string>

namespace aurelis {

using json = nlohmann::json;

struct AurelisConfig {
    LensConfig lens;

    // Add other configs here (AureumConfig, ArcConfig, etc.)

    void save(const std::string& path) const;
    static AurelisConfig load(const std::string& path);
};

inline void to_json(json& j, const LensConfig& cfg) {
    j = json{
        {"vocab_size", cfg.vocab_size},
        {"D", cfg.D},
        {"d_model", cfg.d_model},
        {"d_tau", cfg.d_tau},
        {"d_ff", cfg.d_ff},
        {"num_layers", cfg.num_layers},
        {"num_scales", cfg.num_scales},
        {"lambda_stab", cfg.lambda_stab},
        {"lambda_aux", cfg.lambda_aux},
        {"lr", cfg.lr}
    };
}

inline void from_json(const json& j, LensConfig& cfg) {
    j.at("vocab_size").get_to(cfg.vocab_size);
    j.at("D").get_to(cfg.D);
    j.at("d_model").get_to(cfg.d_model);
    j.at("d_tau").get_to(cfg.d_tau);
    j.at("d_ff").get_to(cfg.d_ff);
    j.at("num_layers").get_to(cfg.num_layers);
    j.at("num_scales").get_to(cfg.num_scales);
    j.at("lambda_stab").get_to(cfg.lambda_stab);
    j.at("lambda_aux").get_to(cfg.lambda_aux);
    j.at("lr").get_to(cfg.lr);
}

inline void AurelisConfig::save(const std::string& path) const {
    json j;
    to_json(j["lens"], lens);
    std::ofstream ofs(path);
    ofs << j.dump(4);
}

inline AurelisConfig AurelisConfig::load(const std::string& path) {
    std::ifstream ifs(path);
    json j = json::parse(ifs);
    AurelisConfig cfg;
    from_json(j["lens"], cfg.lens);
    return cfg;
}

} // namespace aurelis
