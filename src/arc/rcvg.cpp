#include "aurelis/arc/rcvg.hpp"
#include "aurelis/lens/activations.hpp"
#include <cmath>

namespace aurelis::arc {

RCVG::RCVG(const ArcConfig& cfg)
    : cfg(cfg),
      m_W_alpha(aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dm, cfg.num_modes)
{
}

void RCVG::init() {
    m_W_alpha.init_xavier();
}

void RCVG::compute_alpha(const float* m_prev, const float* alpha_prompt, float* alpha) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // Compute alpha^meta = W_alpha @ m_prev + b_alpha
    std::vector<int64_t> shape_m = {static_cast<int64_t>(part.Dm)};
    aurelis::Tensor m_tensor = aurelis::Tensor::from_data(shape_m, m_prev, false);
    aurelis::Tensor alpha_meta = m_W_alpha.forward(m_tensor);

    // Add prompt and softmax
    float max_logit = alpha_meta.data()[0] + (alpha_prompt ? alpha_prompt[0] : 0.0f);
    for (int j = 1; j < cfg.num_modes; ++j) {
        float val = alpha_meta.data()[j] + (alpha_prompt ? alpha_prompt[j] : 0.0f);
        if (val > max_logit) max_logit = val;
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < cfg.num_modes; ++j) {
        float val = alpha_meta.data()[j] + (alpha_prompt ? alpha_prompt[j] : 0.0f);
        sum_exp += std::exp(val - max_logit);
    }
    for (int j = 0; j < cfg.num_modes; ++j) {
        float val = alpha_meta.data()[j] + (alpha_prompt ? alpha_prompt[j] : 0.0f);
        alpha[j] = std::exp(val - max_logit) / sum_exp;
    }
}

float RCVG::compute_epsilon_eff(const float* alpha) const {
    float eps = 0.0f;
    for (int j = 0; j < cfg.num_modes; ++j) {
        eps += alpha[j] * cfg.mode_eps[j];
    }
    return eps;
}

} // namespace aurelis::arc
