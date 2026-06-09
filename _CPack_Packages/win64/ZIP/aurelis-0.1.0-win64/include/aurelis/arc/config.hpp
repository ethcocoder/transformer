#pragma once

#include <aurelis/aureum/config.hpp>

namespace aurelis::arc {

struct ArcConfig {
    aurelis::aureum::AureumConfig aureum_cfg;

    int num_attractors = 64;
    int num_modes = 4; // 0: Deduce, 1: Plan, 2: Create, 3: Analogize

    float tau0 = 1.0f;
    float eta_explore = 0.5f;
    float lambda_sparse = 0.01f;

    // Mode dissipation table (ε̄_j)
    float mode_eps[4] = {0.50f, 0.15f, 0.02f, 0.10f};
};

} // namespace aurelis::arc
