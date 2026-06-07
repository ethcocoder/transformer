#include "aurelis/arc/mc_fws.hpp"
#include <cmath>
#include <cstring>
#include <random>

namespace aurelis::arc {

MCFWS::MCFWS(const ArcConfig& cfg)
    : cfg(cfg), part(StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D)),
      mssp_layout(aurelis::lens::MsspLayout::build(StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D).Dr, 4))
{
}

void MCFWS::init() {
    // Nothing to initialize for now (except mssp_layout is built in constructor)
}

void MCFWS::compute_fw_params(const float* /*c*/, const float* gamma, const float* xi_eff, float epsilon_eff, float alpha_create,
                             float* a_r, float* b_r, int n) const {
    // RNG for INS (Create mode)
    std::mt19937 rng(456);
    std::normal_distribution<float> noise_dist(0.0f, 0.01f);

    for (int t = 0; t < n; ++t) {
        const float* gamma_t = gamma + t * cfg.aureum_cfg.cfg.d_model;
        float* a_t = a_r + t * part.Dr;
        float* b_t = b_r + t * part.Dr;

        for (int i = 0; i < part.Dr; ++i) {
            // CSRG: Use scale index from MSSP to adjust beta_t per scale
            int scale = mssp_layout.scale_index[static_cast<size_t>(i)];
            float scale_factor = 1.0f / std::pow(2.0f, static_cast<float>(scale));
            float beta_t = gamma_t[i % cfg.aureum_cfg.cfg.d_model] * scale_factor;

            // Compute a^r = 1 - ε^eff * β_t
            a_t[i] = 1.0f - epsilon_eff * beta_t;

            // Compute b^r = (1 - a_t[i]) * ξ^eff[i] + γ_t[i % d_model]
            b_t[i] = (1.0f - a_t[i]) * xi_eff[i] + gamma_t[i % cfg.aureum_cfg.cfg.d_model];

            // INS: Inject Gaussian noise only in Create mode (alpha_create is weight)
            if (alpha_create > 0.0f) {
                b_t[i] += alpha_create * noise_dist(rng);
            }
        }
    }
}

} // namespace aurelis::arc
