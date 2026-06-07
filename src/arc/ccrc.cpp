#include "aurelis/arc/ccrc.hpp"
#include <cstring>
#include <cmath>

namespace aurelis::arc {

CCRC::CCRC(const ArcConfig& cfg)
    : cfg(cfg),
      m_W_cr(aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dr, aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dc),
      m_W_mr(aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dm, cfg.num_attractors)
{
}

void CCRC::init() {
    m_W_cr.init_xavier();
    m_W_mr.init_xavier();
}

void CCRC::apply_rcm(const float* c, const float* r, float* c_out) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // c_out = c + W_cr @ r
    std::vector<int64_t> shape_r = {static_cast<int64_t>(part.Dr)};
    aurelis::Tensor r_tensor = aurelis::Tensor::from_data(shape_r, r, false);
    aurelis::Tensor delta_c = m_W_cr.forward(r_tensor);

    for (int i = 0; i < part.Dc; ++i) {
        c_out[i] = c[i] + delta_c.data()[i];
    }
}

float CCRC::apply_erug(float kappa, float epsilon_eff) const {
    return epsilon_eff * (kappa + (1.0f - kappa) * cfg.eta_explore);
}

void CCRC::apply_mrdi(float* log_pi, const float* m) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // log_pi += W_mr @ m
    std::vector<int64_t> shape_m = {static_cast<int64_t>(part.Dm)};
    aurelis::Tensor m_tensor = aurelis::Tensor::from_data(shape_m, m, false);
    aurelis::Tensor delta_logpi = m_W_mr.forward(m_tensor);

    for (int i = 0; i < cfg.num_attractors; ++i) {
        log_pi[i] += delta_logpi.data()[i];
    }
}

} // namespace aurelis::arc
