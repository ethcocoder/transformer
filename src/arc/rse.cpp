#include "aurelis/arc/rse.hpp"
#include "aurelis/scan.h"
#include <cstring>

namespace aurelis::arc {

RSE::RSE(const ArcConfig& cfg) : cfg(cfg) {}

void RSE::init() {}

void RSE::forward_scan(const float* a_r, const float* b_r, int n, float* r) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // Initialize r[0]
    std::memset(r, 0, static_cast<size_t>(part.Dr) * sizeof(float));

    // Scan recurrence: r[t] = a[t] * r[t-1] + b[t]
    for (int t = 1; t < n; ++t) {
        const float* a_t = a_r + t * part.Dr;
        const float* b_t = b_r + t * part.Dr;
        const float* r_prev = r + (t-1)*part.Dr;
        float* r_curr = r + t * part.Dr;

        for (int i = 0; i < part.Dr; ++i) {
            r_curr[i] = a_t[i] * r_prev[i] + b_t[i];
        }
    }
}

} // namespace aurelis::arc
