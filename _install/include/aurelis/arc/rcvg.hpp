#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::arc {

// Reasoning Control Vector Gate (RCVG)
class RCVG {
public:
    explicit RCVG(const ArcConfig& cfg);

    void init();

    // Compute mode vector α
    void compute_alpha(const float* m_prev, const float* alpha_prompt, float* alpha) const;

    // Compute effective dissipation ε^eff
    float compute_epsilon_eff(const float* alpha) const;

    aurelis::lens::Linear& W_alpha() { return m_W_alpha; }

private:
    ArcConfig cfg;
    aurelis::lens::Linear m_W_alpha; // [Dm] -> [num_modes]
};

} // namespace aurelis::arc
