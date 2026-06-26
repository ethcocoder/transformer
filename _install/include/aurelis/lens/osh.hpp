#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/linear.hpp"

namespace aurelis::lens {

class OSH {
 public:
    explicit OSH(const LensConfig& cfg);

    void init();

    void forward_step(const float* c, const float* x_embed, float* logits) const;
    void forward_sequence(const float* c, const float* x_embed, int n,
                          float* logits) const;

    void backward_step(const float* grad_logits, const float* c,
                       const float* x_embed, float* grad_c, float* grad_x,
                       Tensor& grad_Wout, Tensor& grad_bout, Tensor& grad_Wskip,
                       Tensor& grad_bskip) const;

    Linear& out_proj() { return W_out_; }
    Linear& skip_proj() { return W_skip_; }

 private:
    LensConfig cfg_;
    Linear W_out_;
    Linear W_skip_;
};

}  // namespace aurelis::lens
