#include "aurelis/lens/activations.hpp"

#include <algorithm>
#include <cmath>

namespace aurelis::lens {

float silu(float x) {
    if (!std::isfinite(x)) return x < 0 ? 0.0f : x;
    if (x > 80.0f) return x;
    if (x < -80.0f) return 0.0f;
    const float s = 1.0f / (1.0f + std::exp(-x));
    return x * s;
}

float dsilu(float x) {
    if (!std::isfinite(x)) return x < 0 ? 0.0f : 1.0f;
    if (x > 80.0f) return 1.0f;
    if (x < -80.0f) return 0.0f;
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
        if (!std::isfinite(in[i]) || !std::isfinite(grad_out[i])) {
            grad_in[i] = 0.0f;
        } else {
            grad_in[i] = grad_out[i] * dsilu(in[i]);
        }
    }
}

void sigmoid_forward(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(in[i])) {
            out[i] = in[i] < 0 ? 0.0f : 1.0f;
        } else if (in[i] > 80.0f) {
            out[i] = 1.0f;
        } else if (in[i] < -80.0f) {
            out[i] = 0.0f;
        } else {
            out[i] = 1.0f / (1.0f + std::exp(-in[i]));
        }
    }
}

void sigmoid_backward(const float* in, const float* grad_out, float* grad_in,
                      std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(in[i]) || !std::isfinite(grad_out[i])) {
            grad_in[i] = 0.0f;
        } else if (in[i] > 80.0f || in[i] < -80.0f) {
            grad_in[i] = 0.0f;
        } else {
            const float s = 1.0f / (1.0f + std::exp(-in[i]));
            grad_in[i] = grad_out[i] * s * (1.0f - s);
        }
    }
}

}  // namespace aurelis::lens
