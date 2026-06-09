#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/linear.hpp"
#include "aurelis/state_partition.hpp"

namespace aurelis::lens {

struct SpiOutput {
    float aux_loss = 0.0f;
};

class SPI {
 public:
    explicit SPI(const LensConfig& cfg);

    void init();

    /* h[D] -> c[Dc], e[De], r[Dr], m[Dm]; returns L_aux contribution */
    float forward_step(const float* h, float* c, float* e, float* r,
                       float* m) const;

    void forward_sequence(const float* h, float* c, float* e, float* r,
                          float* m, int n, float& aux_loss) const;

    void backward_step(const float* h, const float* grad_c, const float* grad_e,
                       const float* grad_r, const float* grad_m, float* grad_h,
                       Tensor& grad_We, Tensor& grad_be, Tensor& grad_Wr,
                       Tensor& grad_br, Tensor& grad_Wm,
                       Tensor& grad_bm) const;

    Linear& W_e() { return W_e_; }
    Linear& W_r() { return W_r_; }
    Linear& W_m() { return W_m_; }

    StatePartition partition() const { return part_; }

 private:
    LensConfig cfg_;
    StatePartition part_;
    Linear W_e_;
    Linear W_r_;
    Linear W_m_;
};

}  // namespace aurelis::lens
