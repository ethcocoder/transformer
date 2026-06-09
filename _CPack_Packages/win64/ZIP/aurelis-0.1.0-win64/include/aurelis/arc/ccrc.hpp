#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::arc {

// Cross-Channel Reasoning Core (CCRC)
class CCRC {
public:
    explicit CCRC(const ArcConfig& cfg);

    void init();

    // RCM: Reasoning-to-Content Modulation
    void apply_rcm(const float* c, const float* r, float* c_out) const;

    // ERUG: Epistemic Reasoning Uncertainty Gating
    float apply_erug(float kappa, float epsilon_eff) const;

    // MRDI: Meta-Reasoning Diffusion Injection
    void apply_mrdi(float* log_pi, const float* m) const;

    aurelis::lens::Linear& W_cr() { return m_W_cr; }
    aurelis::lens::Linear& W_mr() { return m_W_mr; }

private:
    ArcConfig cfg;
    aurelis::lens::Linear m_W_cr; // [Dr] -> [Dc]
    aurelis::lens::Linear m_W_mr; // [Dm] -> [num_attractors]
};

} // namespace aurelis::arc
