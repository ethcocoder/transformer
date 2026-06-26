#pragma once

#include <aurelis/aureum/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::aureum {

// Epistemic Self-Encoding (ESE)
class ESE {
public:
    explicit ESE(const AureumConfig& cfg);

    void init();

    // Forward for single step (for sequential use)
    float forward_step(const float* c_t, const float* m_prev, const float* gate, float* e_t) const;

    // Forward for entire sequence using scan
    void forward_sequence(const float* c, const float* m, const float* gates, int n, float* e) const;

    // Backward pass
    void backward_step(const float* c_t, const float* m_prev, const float* gate, const float* grad_e,
                       float* grad_c, float* grad_m,
                       aurelis::Tensor& grad_W_ec, aurelis::Tensor& grad_b_ec,
                       aurelis::Tensor& grad_W_sigma, aurelis::Tensor& grad_b_sigma) const;

    aurelis::lens::Linear& W_ec() { return W_ec_; }
    aurelis::lens::Linear& W_sigma() { return W_sigma_; }

private:
    AureumConfig cfg_;
    aurelis::lens::Linear W_ec_;  // c -> e injection
    aurelis::lens::Linear W_sigma_; // e -> sigma (UVH)
};

} // namespace aurelis::aureum
