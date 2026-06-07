#include "aurelis/arc/ask_r.hpp"
#include <cstring>
#include <cmath>

namespace aurelis::arc {

ArcPassScheduler::ArcPassScheduler(const ArcConfig& cfg)
    : cfg(cfg),
      _ale(cfg),
      _rcvg(cfg),
      _mc_fws(cfg),
      _rse(cfg),
      _ccrc(cfg),
      _roi(cfg),
      _aureum(cfg.aureum_cfg)
{
}

void ArcPassScheduler::init() {
    _ale.init();
    _rcvg.init();
    _mc_fws.init();
    _rse.init();
    _ccrc.init();
    _roi.init();
    _aureum.init();
}

void ArcPassScheduler::forward(const float* c_lens, const float* gamma_lens, const float* x_embed,
                               int n, float* e, float* r, float* m, float* logits, float* aux_loss) {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.aureum_cfg.cfg.D);

    // --- Pass 1-2: AUREUM to get e and (placeholders for m) ---
    std::vector<float> kappa(static_cast<size_t>(n));
    std::vector<float> c_mod(static_cast<size_t>(n * cfg.aureum_cfg.cfg.d_model));
    _aureum.forward(c_lens, n, e, kappa.data(), m, c_mod.data());

    // --- Pass 3: Compute r ---
    std::vector<float> alpha_prompt(cfg.num_modes, 0.0f);
    std::vector<float> a_r(static_cast<size_t>(n * part.Dr));
    std::vector<float> b_r(static_cast<size_t>(n * part.Dr));
    std::vector<float> pi(cfg.num_attractors);
    std::vector<float> xi_eff(static_cast<size_t>(part.Dr));

    float reason_entropy_sum = 0.0f;

    std::vector<float> alpha(cfg.num_modes);
    for (int t = 0; t < n; ++t) {
        const float* m_prev = (t > 0) ? (m + (t-1)*part.Dm) : nullptr;
        if (t == 0) {
            std::vector<float> m0(static_cast<size_t>(part.Dm), 0.0f);
            m0[0] = 1.0f;
            _rcvg.compute_alpha(m0.data(), alpha_prompt.data(), alpha.data());
        } else {
            _rcvg.compute_alpha(m_prev, alpha_prompt.data(), alpha.data());
        }

        // Select attractor (ASA)
        _ale.select_attractor(c_lens + t*cfg.aureum_cfg.cfg.d_model, e + t*part.De, alpha.data(), pi.data());

        // Apply MRDI: log_pi += W_mr @ m_prev
        std::vector<float> log_pi(static_cast<size_t>(cfg.num_attractors));
        for (int i = 0; i < cfg.num_attractors; ++i) {
            log_pi[i] = std::log(pi[i] + 1e-12f); // add epsilon to avoid log(0)
        }
        if (t > 0) {
            _ccrc.apply_mrdi(log_pi.data(), m_prev);
        }

        // Recompute pi from modified log_pi
        float max_logit = log_pi[0];
        for (int i = 1; i < cfg.num_attractors; ++i) {
            if (log_pi[i] > max_logit) max_logit = log_pi[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < cfg.num_attractors; ++i) {
            sum_exp += std::exp(log_pi[i] - max_logit);
        }
        for (int i = 0; i < cfg.num_attractors; ++i) {
            pi[i] = std::exp(log_pi[i] - max_logit) / sum_exp;
        }

        // Compute TEM (reasoning entropy)
        float h_reason = 0.0f;
        for (int i = 0; i < cfg.num_attractors; ++i) {
            h_reason -= pi[i] * std::log(pi[i] + 1e-12f);
        }
        reason_entropy_sum += h_reason;

        // Compute effective attractor (LAB)
        _ale.compute_effective_attractor(pi.data(), xi_eff.data());

        float eps_eff = _rcvg.compute_epsilon_eff(alpha.data());
        float eps_erug = _ccrc.apply_erug(kappa[t], eps_eff);

        // Compute MCFWS params (using alpha[2] which is Create mode)
        _mc_fws.compute_fw_params(c_lens + t*cfg.aureum_cfg.cfg.d_model, gamma_lens, xi_eff.data(), eps_erug, alpha[2],
                                  a_r.data() + t*part.Dr, b_r.data() + t*part.Dr, 1);
    }
    _rse.forward_scan(a_r.data(), b_r.data(), n, r);

    // --- Pass 4: Update m with AUREUM's KMG (using c and e) ---
    _aureum.kmg().forward_sequence(c_lens, e, n, m);

    // --- Compute logits with ROI ---
    std::vector<float> c_out(static_cast<size_t>(n * part.Dc));
    for (int t = 0; t < n; ++t) {
        _ccrc.apply_rcm(c_lens + t*cfg.aureum_cfg.cfg.d_model, r + t*part.Dr, c_out.data() + t*part.Dc);
    }
    _roi.compute_logits(c_out.data(), x_embed, n, logits);

    if (aux_loss) {
        // Add sparse loss for pi (lambda_sparse * entropy)
        float avg_reason_entropy = reason_entropy_sum / static_cast<float>(n);
        *aux_loss = cfg.lambda_sparse * avg_reason_entropy;
    }
}

} // namespace aurelis::arc
