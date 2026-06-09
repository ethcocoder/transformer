#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/ptk.hpp"
#include "aurelis/tensor.hpp"

namespace aurelis::lens {

class CSC {
 public:
    explicit CSC(const LensConfig& cfg);

    void init();

    /* a,b: [n,D] -> h_raw[n,D] (optional), h_out[n,D] after CCM */
    void forward(const float* a, const float* b, int n, float* h_out,
                 float* h_raw = nullptr) const;

    void backward(const float* a, const float* h_raw, const float* grad_h_out,
                  int n, float* grad_a, float* grad_b, float* grad_h_raw,
                  Tensor& grad_M, Tensor& grad_c_bias) const;

    Tensor& mix_matrix() { return M_; }
    Tensor& mix_bias() { return c_bias_; }

 private:
    LensConfig cfg_;
    Tensor M_;
    Tensor c_bias_;
};

}  // namespace aurelis::lens
