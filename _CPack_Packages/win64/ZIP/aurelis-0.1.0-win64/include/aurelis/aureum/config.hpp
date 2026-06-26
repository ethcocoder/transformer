#pragma once

#include <aurelis/lens/config.hpp>

namespace aurelis::aureum {

struct AureumConfig {
    aurelis::lens::LensConfig cfg;

    // Additional AUREUM-specific hyperparameters
    float lambda_calibration = 0.1f;
    float lambda_mcl = 0.05f;
    float lambda_mrl = 0.01f;
    float lambda_mcer = 0.001f;
    float tau_ood = 0.5f;
    float lambda_ood = 10.0f;
    float eps_gm = 1e-6f;
    int num_domains = 10;
};

} // namespace aurelis::aureum
