#include "aurelis/arc/ale.hpp"
#include "aurelis/lens/activations.hpp"
#include <random>
#include <cmath>
#include <cstring>

namespace aurelis::arc {

ALE::ALE(const ArcConfig& cfg)
    : cfg(cfg),
      Xi(static_cast<size_t>(cfg.num_attractors * aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dr), 0.0f),
      Lambda(static_cast<size_t>(cfg.num_attractors * aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dr * aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dr), 0.0f),
      m_W_pi(aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dc + aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).De + cfg.num_modes, cfg.num_attractors)
{
}

void ALE::init() {
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // Initialize attractor bank
    for (size_t i = 0; i < Xi.size(); ++i) {
        Xi[i] = dist(rng);
    }

    // Initialize Lambda (per-attractor identity + eps)
    for (int m = 0; m < cfg.num_attractors; ++m) {
        for (int i = 0; i < part.Dr; ++i) {
            size_t idx = static_cast<size_t>(m * part.Dr * part.Dr + i * part.Dr + i);
            Lambda[idx] = 1.0f + cfg.aureum_cfg.eps_gm;
        }
    }

    m_W_pi.init_xavier();
}

void ALE::select_attractor(const float* c, const float* e, const float* alpha, float* pi) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // Concatenate [c; e; alpha]
    const int input_dim = part.Dc + part.De + cfg.num_modes;
    std::vector<float> concat(static_cast<size_t>(input_dim));
    std::memcpy(concat.data(), c, static_cast<size_t>(part.Dc) * sizeof(float));
    std::memcpy(concat.data() + part.Dc, e, static_cast<size_t>(part.De) * sizeof(float));
    std::memcpy(concat.data() + part.Dc + part.De, alpha, static_cast<size_t>(cfg.num_modes) * sizeof(float));

    // Forward through W_pi
    std::vector<int64_t> shape_input = {static_cast<int64_t>(input_dim)};
    aurelis::Tensor input_tensor = aurelis::Tensor::from_data(shape_input, concat.data(), false);
    aurelis::Tensor logits_tensor = m_W_pi.forward(input_tensor);

    // Softmax to get pi
    const float* logits = logits_tensor.data();
    float max_logit = logits[0];
    for (int m = 1; m < cfg.num_attractors; ++m) {
        if (logits[m] > max_logit) max_logit = logits[m];
    }
    float sum_exp = 0.0f;
    for (int m = 0; m < cfg.num_attractors; ++m) {
        sum_exp += std::exp(logits[m] - max_logit);
    }
    for (int m = 0; m < cfg.num_attractors; ++m) {
        pi[m] = std::exp(logits[m] - max_logit) / sum_exp;
    }
}

void ALE::get_basin_precision(int m, float* Lm) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);
    size_t start = static_cast<size_t>(m * part.Dr * part.Dr);
    std::memcpy(Lm, Lambda.data() + start, static_cast<size_t>(part.Dr * part.Dr) * sizeof(float));
}

void ALE::compute_effective_attractor(const float* pi, float* xi_eff) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);
    std::memset(xi_eff, 0, static_cast<size_t>(part.Dr) * sizeof(float));
    for (int m = 0; m < cfg.num_attractors; ++m) {
        const float* xi_m = Xi.data() + static_cast<size_t>(m * part.Dr);
        float p = pi[m];
        for (int i = 0; i < part.Dr; ++i) {
            xi_eff[i] += p * xi_m[i];
        }
    }
}

} // namespace aurelis::arc
