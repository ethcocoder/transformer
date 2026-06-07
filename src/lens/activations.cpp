#include "aurelis/lens/activations.hpp"

#include <cmath>

namespace aurelis::lens {

float silu(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return x * s;
}

float dsilu(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

void silu_forward(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = silu(in[i]);
    }
}

void silu_backward(const float* in, const float* grad_out, float* grad_in,
                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        grad_in[i] = grad_out[i] * dsilu(in[i]);
    }
}

void sigmoid_forward(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void sigmoid_backward(const float* in, const float* grad_out, float* grad_in,
                      std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float s = 1.0f / (1.0f + std::exp(-in[i]));
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

}  // namespace aurelis::lens
