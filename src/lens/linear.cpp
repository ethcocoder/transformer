#include "aurelis/lens/linear.hpp"

#include "aurelis/matmul.h"

#include <cmath>
#include <cstring>
#include <random>

namespace aurelis::lens {

Linear::Linear(int in_features, int out_features)
    : in_(in_features),
      out_(out_features),
      W_(Tensor::zeros({out_features, in_features}, true)),
      b_(Tensor::zeros({out_features}, true)) {}

void Linear::init_xavier() {
    std::mt19937 rng(42);
    const float scale =
        std::sqrt(2.0f / static_cast<float>(in_ + out_));
    std::normal_distribution<float> dist(0.0f, scale);
    for (int64_t i = 0; i < W_.numel(); ++i) {
        W_.at(i) = dist(rng);
    }
    b_.fill(0.0f);
}

void linear_forward(const float* W, const float* b, const float* x, float* y,
                    int out_f, int in_f) {
    aurelis_matmul_f32(1.0f, W, x, 0.0f, y, static_cast<size_t>(out_f),
                       static_cast<size_t>(in_f), 1);
    for (int i = 0; i < out_f; ++i) {
        y[i] += b[i];
    }
}

void linear_backward(const float* W, const float* x, const float* grad_y,
                     float* grad_W, float* grad_b, float* grad_x, int out_f,
                     int in_f) {
    for (int o = 0; o < out_f; ++o) {
        grad_b[o] += grad_y[o];
        for (int i = 0; i < in_f; ++i) {
            grad_W[o * in_f + i] += grad_y[o] * x[i];
            grad_x[i] += W[o * in_f + i] * grad_y[o];
        }
    }
}

Tensor Linear::forward(const Tensor& x) const {
    Tensor y = Tensor::zeros({out_});
    linear_forward(W_.data(), b_.data(), x.data(), y.data(), out_, in_);
    return y;
}

void Linear::backward(const Tensor& grad_out, const Tensor& x, Tensor& grad_W,
                      Tensor& grad_b, Tensor& grad_x) const {
    linear_backward(W_.data(), x.data(), grad_out.data(), grad_W.data(),
                    grad_b.data(), grad_x.data(), out_, in_);
}

}  // namespace aurelis::lens
