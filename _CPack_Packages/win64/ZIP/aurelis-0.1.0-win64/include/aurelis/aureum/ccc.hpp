#pragma once

#include <aurelis/aureum/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::aureum {

// Content-Confidence Calibration (CCC)
class CCC {
public:
    explicit CCC(const AureumConfig& cfg);

    void init();

    // ECM: Epistemic Content Modulation
    void modulate_content(const float* c, float kappa, const float* e, float* c_mod) const;

    // MEG: Meta-Envelope Gating
    void gate_meta(const float* m_prev, float* gate) const;

    aurelis::lens::Linear& W_fallback() { return W_fallback_; }
    aurelis::lens::Linear& W_me() { return W_me_; }

private:
    AureumConfig cfg_;
    aurelis::lens::Linear W_fallback_;
    aurelis::lens::Linear W_me_;
};

} // namespace aurelis::aureum
