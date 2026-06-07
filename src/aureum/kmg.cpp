#include "aurelis/aureum/kmg.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::aureum {

KMG::KMG(const AureumConfig& cfg)
    : cfg_(cfg),
      W_Lm_(cfg.cfg.d_model + aurelis::StatePartition::from_total_dim(cfg.cfg.D).De,
            aurelis::StatePartition::from_total_dim(cfg.cfg.D).Dm) {}

void KMG::init() {
    W_Lm_.init_xavier();
}

void KMG::compute_Gm(const float* Lm, float* Gm, int Dm) const {
    // Gm = Lm * Lm^T + eps*I
    for (int i = 0; i < Dm; ++i) {
        for (int j = 0; j < Dm; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < Dm; ++k) {
                sum += Lm[i*Dm + k] * Lm[j*Dm + k];
            }
            Gm[i*Dm + j] = sum;
        }
        Gm[i*Dm + i] += cfg_.eps_gm;
    }
}

float KMG::geodesic_distance(const float* d1, const float* d2, const float* Gm, int Dm) const {
    // (d1-d2)^T Gm (d1-d2)
    std::vector<float> diff(static_cast<std::size_t>(Dm));
    for (int i = 0; i < Dm; ++i) {
        diff[static_cast<std::size_t>(i)] = d1[i] - d2[i];
    }

    std::vector<float> Gm_diff(static_cast<std::size_t>(Dm), 0.0f);
    for (int i = 0; i < Dm; ++i) {
        for (int j = 0; j < Dm; ++j) {
            Gm_diff[static_cast<std::size_t>(i)] += Gm[i*Dm + j] * diff[static_cast<std::size_t>(j)];
        }
    }

    float dist_sq = 0.0f;
    for (int i = 0; i < Dm; ++i) {
        dist_sq += diff[static_cast<std::size_t>(i)] * Gm_diff[static_cast<std::size_t>(i)];
    }
    return std::sqrt(dist_sq);
}

void KMG::forward_step(const float* c_t, const float* e_t, float* m_t) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);

    // Concatenate c and e
    const int D_concat = cfg_.cfg.d_model + part.De;
    std::vector<float> concat(static_cast<std::size_t>(D_concat));
    std::memcpy(concat.data(), c_t, static_cast<std::size_t>(cfg_.cfg.d_model) * sizeof(float));
    std::memcpy(concat.data() + cfg_.cfg.d_model, e_t, static_cast<std::size_t>(part.De) * sizeof(float));

    std::vector<int64_t> shape_concat = {static_cast<int64_t>(D_concat)};
    aurelis::Tensor concat_tensor = aurelis::Tensor::from_data(shape_concat, concat.data(), false);
    aurelis::Tensor m_out = W_Lm_.forward(concat_tensor);

    std::memcpy(m_t, m_out.data(), static_cast<std::size_t>(part.Dm) * sizeof(float));
}

void KMG::forward_sequence(const float* c, const float* e, int n, float* m) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    for (int t = 0; t < n; ++t) {
        forward_step(c + t*cfg_.cfg.d_model, e + t*part.De, m + t*part.Dm);
    }
}

} // namespace aurelis::aureum
