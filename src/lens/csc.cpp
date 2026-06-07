#include "aurelis/lens/csc.hpp"

#include "aurelis/matmul.h"

#include <cstring>
#include <vector>

namespace aurelis::lens {

CSC::CSC(const LensConfig& cfg)
    : cfg_(cfg),
      M_(Tensor::zeros({cfg.D, cfg.D}, true)),
      c_bias_(Tensor::zeros({cfg.D}, true)) {}

void CSC::init() {
    for (int i = 0; i < cfg_.D; ++i) {
        for (int j = 0; j < cfg_.D; ++j) {
            M_.at(i * cfg_.D + j) = (i == j) ? 1.0f : 0.0f;
        }
    }
    c_bias_.fill(0.0f);
}

void CSC::forward(const float* a, const float* b, int n, float* h_out,
                  float* h_raw_out) const {
    std::vector<float> h0(static_cast<size_t>(cfg_.D), 0.0f);
    std::vector<float> h_raw_local;
    float* h_raw = h_raw_out;
    if (!h_raw) {
        h_raw_local.resize(static_cast<size_t>(n * cfg_.D));
        h_raw = h_raw_local.data();
    }
    parallel_scan_forward(a, b, h0.data(), h_raw, n, cfg_.D);

    for (int t = 0; t < n; ++t) {
        aurelis_matmul_f32(1.0f, M_.data(), h_raw + t * cfg_.D, 0.0f,
                           h_out + t * cfg_.D, static_cast<size_t>(cfg_.D),
                           static_cast<size_t>(cfg_.D), 1);
        for (int d = 0; d < cfg_.D; ++d) {
            h_out[t * cfg_.D + d] += c_bias_.at(d);
        }
    }
}

void CSC::backward(const float* a, const float* h_raw, const float* grad_h_out,
                   int n, float* grad_a, float* grad_b, float* grad_h_raw,
                   Tensor& grad_M, Tensor& grad_c_bias) const {
    std::memset(grad_h_raw, 0, static_cast<size_t>(n * cfg_.D) * sizeof(float));
    std::memset(grad_a, 0, static_cast<size_t>(n * cfg_.D) * sizeof(float));
    std::memset(grad_b, 0, static_cast<size_t>(n * cfg_.D) * sizeof(float));

    for (int t = 0; t < n; ++t) {
        for (int d = 0; d < cfg_.D; ++d) {
            grad_c_bias.at(d) += grad_h_out[t * cfg_.D + d];
        }
        for (int i = 0; i < cfg_.D; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < cfg_.D; ++j) {
                sum += M_.at(j * cfg_.D + i) * grad_h_out[t * cfg_.D + j];
                grad_M.at(i * cfg_.D + j) +=
                    grad_h_out[t * cfg_.D + i] * h_raw[t * cfg_.D + j];
            }
            grad_h_raw[t * cfg_.D + i] = sum;
        }
    }

    std::vector<float> h0(static_cast<size_t>(cfg_.D), 0.0f);
    std::vector<float> grad_h0(static_cast<size_t>(cfg_.D), 0.0f);
    parallel_scan_backward(a, h_raw, h0.data(), grad_h_raw, grad_a, grad_b,
                           grad_h0.data(), n, cfg_.D);
}

}  // namespace aurelis::lens
