#include "aurelis/aureum/ccc.hpp"
#include "aurelis/lens/activations.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::aureum {

CCC::CCC(const AureumConfig& cfg)
    : cfg_(cfg),
      W_fallback_(aurelis::StatePartition::from_total_dim(cfg.cfg.D).De, cfg.cfg.d_model),
      W_me_(aurelis::StatePartition::from_total_dim(cfg.cfg.D).Dm, aurelis::StatePartition::from_total_dim(cfg.cfg.D).De) {}

void CCC::init() {
    W_fallback_.init_xavier();
    W_me_.init_xavier();
}

void CCC::modulate_content(const float* c, float kappa, const float* e, float* c_mod) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);

    // Compute fallback: W_fallback @ e
    std::vector<int64_t> shape_e = {static_cast<int64_t>(part.De)};
    aurelis::Tensor e_tensor = aurelis::Tensor::from_data(shape_e, e, false);
    aurelis::Tensor fallback = W_fallback_.forward(e_tensor);

    // ECM: c_mod = kappa * c + (1-kappa) * fallback
    for (int i = 0; i < cfg_.cfg.d_model; ++i) {
        c_mod[i] = kappa * c[i] + (1.0f - kappa) * fallback.at(i);
    }
}

void CCC::gate_meta(const float* m_prev, float* gate) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    std::vector<int64_t> shape_m = {static_cast<int64_t>(part.Dm)};
    aurelis::Tensor m_tensor = aurelis::Tensor::from_data(shape_m, m_prev, false);
    aurelis::Tensor gate_out = W_me_.forward(m_tensor);
    aurelis::lens::sigmoid_forward(gate_out.data(), gate, static_cast<std::size_t>(gate_out.numel()));
}

} // namespace aurelis::aureum
