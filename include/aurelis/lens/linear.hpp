#pragma once

#include "aurelis/tensor.hpp"

#include <vector>

namespace aurelis::lens {

class Linear {
 public:
    Linear() = default;
    Linear(int in_features, int out_features);

    void init_xavier();

    Tensor forward(const Tensor& x) const;
    void backward(const Tensor& grad_out, const Tensor& x, Tensor& grad_W,
                  Tensor& grad_b, Tensor& grad_x) const;

    Tensor& weight() { return W_; }
    Tensor& bias() { return b_; }
    const Tensor& weight() const { return W_; }
    const Tensor& bias() const { return b_; }

    int in_features() const { return in_; }
    int out_features() const { return out_; }

    bool save(const std::string& filepath) const;
    static Linear load(const std::string& filepath);

 private:
    int in_ = 0;
    int out_ = 0;
    Tensor W_;
    Tensor b_;
};

void linear_forward(const float* W, const float* b, const float* x, float* y,
                    int out_f, int in_f);
void linear_backward(const float* W, const float* x, const float* grad_y,
                     float* grad_W, float* grad_b, float* grad_x, int out_f,
                     int in_f);

}  // namespace aurelis::lens
