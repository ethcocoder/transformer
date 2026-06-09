#pragma once

#include <aurelis/aureum/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::aureum {

// Competence Quotient (CQ)
class CQ {
public:
    explicit CQ(const AureumConfig& cfg);

    void init();

    // Forward: e_t -> kappa_t
    float forward_step(const float* e_t) const;

    void forward_sequence(const float* e, int n, float* kappa) const;

    // Backward
    void backward_step(const float* e_t, float grad_kappa,
                      float* grad_e,
                      aurelis::Tensor& grad_W_kappa, aurelis::Tensor& grad_b_kappa) const;

    aurelis::lens::Linear& W_kappa() { return W_kappa_; }

private:
    AureumConfig cfg_;
    aurelis::lens::Linear W_kappa_;
};

} // namespace aurelis::aureum
