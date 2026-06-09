#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::arc {

// Reasoning Output Interface (ROI)
class ROI {
public:
    explicit ROI(const ArcConfig& cfg);

    void init();

    // TEM: Temperature from reasoning
    float compute_temperature(const float* r) const;

    // RCLR: Reasoning-conditioned logit readout
    void compute_logits(const float* c_out, const float* x_embed, int n, float* logits) const;

    aurelis::lens::Linear& W_tau() { return m_W_tau; }
    aurelis::lens::Linear& W_out() { return m_W_out; }
    aurelis::lens::Linear& W_skip() { return m_W_skip; }

private:
    ArcConfig cfg;
    aurelis::lens::Linear m_W_tau; // [Dr] -> [1] (pre-sigmoid temperature)
    aurelis::lens::Linear m_W_out; // [Dc] -> [vocab_size]
    aurelis::lens::Linear m_W_skip; // [d_model] -> [vocab_size]
};

} // namespace aurelis::arc
