#include "aurelis/aureum/pii.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::aureum {

AureumPassScheduler::AureumPassScheduler(const AureumConfig& cfg)
    : cfg_(cfg),
      ese_(cfg),
      cq_(cfg),
      kmg_(cfg),
      ccc_(cfg) {}

void AureumPassScheduler::init() {
    ese_.init();
    cq_.init();
    kmg_.init();
    ccc_.init();
}

void AureumPassScheduler::forward(const float* c_lens, int n,
                                  float* e, float* kappa, float* m, float* c_mod) {
    // Pass 1: compute e using c and m_prev
    pass1(c_lens, n, e, m);

    // Pass 2: compute kappa and m
    cq_.forward_sequence(e, n, kappa);
    pass2(c_lens, e, n, m);

    // Pass 3: compute c_mod
    pass3(c_lens, e, kappa, n, c_mod);
}

void AureumPassScheduler::pass1(const float* c, int n, float* e, const float* m) {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    std::vector<float> gates(static_cast<size_t>(n * part.De));
    for (int t = 0; t < n; ++t) {
        const float* m_prev = (t > 0) ? (m + (t-1)*part.Dm) : nullptr;
        float* gate_t = gates.data() + t*part.De;
        if (m_prev != nullptr) {
            ccc_.gate_meta(m_prev, gate_t);
        } else {
            std::fill(gate_t, gate_t + part.De, 1.0f);
        }
    }
    ese_.forward_sequence(c, m, gates.data(), n, e);
}

void AureumPassScheduler::pass2(const float* c, const float* e, int n, float* m) {
    kmg_.forward_sequence(c, e, n, m);
}

void AureumPassScheduler::pass3(const float* c, const float* e, const float* kappa, int n, float* c_mod) {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    for (int t = 0; t < n; ++t) {
        ccc_.modulate_content(c + t*cfg_.cfg.d_model, kappa[t], e + t*part.De, c_mod + t*cfg_.cfg.d_model);
    }
}

} // namespace aurelis::aureum
