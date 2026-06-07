#include "aurelis/arc/roi.hpp"
#include <cmath>
#include <cstring>

namespace aurelis::arc {

ROI::ROI(const ArcConfig& cfg)
    : cfg(cfg),
      m_W_tau(aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dr, 1),
      m_W_out(aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dc, cfg.aureum_cfg.cfg.vocab_size),
      m_W_skip(cfg.aureum_cfg.cfg.d_model, cfg.aureum_cfg.cfg.vocab_size)
{
}

void ROI::init() {
    m_W_tau.init_xavier();
    m_W_out.init_xavier();
    m_W_skip.init_xavier();
}

float ROI::compute_temperature(const float* r) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    std::vector<int64_t> shape_r = {static_cast<int64_t>(part.Dr)};
    aurelis::Tensor r_tensor = aurelis::Tensor::from_data(shape_r, r, false);
    aurelis::Tensor logit_tau = m_W_tau.forward(r_tensor);

    float sigmoid_val = 1.0f / (1.0f + std::exp(-logit_tau.data()[0]));
    return cfg.tau0 * sigmoid_val;
}

void ROI::compute_logits(const float* c_out, const float* x_embed, int n, float* logits) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);
    const int vocab_size = cfg.aureum_cfg.cfg.vocab_size;
    const int d_model = cfg.aureum_cfg.cfg.d_model;

    for (int t = 0; t < n; ++t) {
        const float* c_t = c_out + t * part.Dc;
        const float* x_t = x_embed + t * d_model;
        float* logits_t = logits + t * vocab_size;

        std::vector<int64_t> shape_c = {static_cast<int64_t>(part.Dc)};
        aurelis::Tensor c_tensor = aurelis::Tensor::from_data(shape_c, c_t, false);
        aurelis::Tensor o = m_W_out.forward(c_tensor);

        std::vector<int64_t> shape_x = {static_cast<int64_t>(d_model)};
        aurelis::Tensor x_tensor = aurelis::Tensor::from_data(shape_x, x_t, false);
        aurelis::Tensor s = m_W_skip.forward(x_tensor);

        for (int i = 0; i < vocab_size; ++i) {
            logits_t[i] = o.at(i) + s.at(i);
        }
    }
}

} // namespace aurelis::arc
