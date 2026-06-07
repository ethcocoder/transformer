#include "aurelis/lens/osh.hpp"

#include <cstring>

namespace aurelis::lens {

OSH::OSH(const LensConfig& cfg)
    : cfg_(cfg),
      W_out_(cfg.d_model, cfg.vocab_size),
      W_skip_(cfg.d_model, cfg.vocab_size) {}

void OSH::init() {
    W_out_.init_xavier();
    W_skip_.init_xavier();
}

void OSH::forward_step(const float* c, const float* x_embed,
                       float* logits) const {
    Tensor c_t = Tensor::from_data({cfg_.d_model}, c, false);
    Tensor x_t = Tensor::from_data({cfg_.d_model}, x_embed, false);
    Tensor o = W_out_.forward(c_t);
    Tensor s = W_skip_.forward(x_t);
    for (int i = 0; i < cfg_.vocab_size; ++i) {
        logits[i] = o.at(i) + s.at(i);
    }
}

void OSH::forward_sequence(const float* c, const float* x_embed, int n,
                           float* logits) const {
    for (int t = 0; t < n; ++t) {
        forward_step(c + t * cfg_.d_model, x_embed + t * cfg_.d_model,
                     logits + t * cfg_.vocab_size);
    }
}

void OSH::backward_step(const float* grad_logits, const float* c,
                        const float* x_embed, float* grad_c, float* grad_x,
                        Tensor& grad_Wout, Tensor& grad_bout,
                        Tensor& grad_Wskip, Tensor& grad_bskip) const {
    Tensor gl = Tensor::from_data({cfg_.vocab_size}, grad_logits, false);
    Tensor c_t = Tensor::from_data({cfg_.d_model}, c, false);
    Tensor x_t = Tensor::from_data({cfg_.d_model}, x_embed, false);
    Tensor gc = Tensor::zeros({cfg_.d_model});
    Tensor gx = Tensor::zeros({cfg_.d_model});
    W_out_.backward(gl, c_t, grad_Wout, grad_bout, gc);
    W_skip_.backward(gl, x_t, grad_Wskip, grad_bskip, gx);
    for (int i = 0; i < cfg_.d_model; ++i) {
        grad_c[i] += gc.at(i);
        grad_x[i] += gx.at(i);
    }
}

}  // namespace aurelis::lens
