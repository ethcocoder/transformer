#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/tensor.hpp"

namespace aurelis::lens {

class MGP {
 public:
    explicit MGP(const LensConfig& cfg);

    void init();

    void forward(const float* h_in, float* h_out) const;
    void forward_sequence(const float* h_in, float* h_out, int n) const;

    void backward_step(const float* h_in, const float* grad_out,
                       float* grad_in, Tensor& grad_mu, Tensor& grad_L) const;

    Tensor& mu() { return mu_; }
    Tensor& L() { return L_; }

 private:
    LensConfig cfg_;
    Tensor mu_;
    Tensor L_;
};

}  // namespace aurelis::lens
