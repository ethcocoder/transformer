#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/linear.hpp"
#include "aurelis/tensor.hpp"

#include <vector>

namespace aurelis::lens {

class IETCF {
 public:
    explicit IETCF(const LensConfig& cfg);

    void init();

    /* tokens[n] -> x_embed[n*d_model], gamma[n*d_model] */
    void forward(const int* tokens, int n, float* x_embed, float* gamma);

    Tensor& embedding() { return E_; }
    Linear& gamma_proj() { return W_gamma_; }
    Tensor& rotation() { return R_; }

 private:
    LensConfig cfg_;
    Tensor E_;
    Tensor R_;
    Linear W_gamma_;
    std::vector<float> tau_;
};

}  // namespace aurelis::lens
