#include "aurelis/lens/mgp.hpp"

#include "aurelis/linalg.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::lens {

MGP::MGP(const LensConfig& cfg)
    : cfg_(cfg), mu_(Tensor::zeros({cfg.D}, true)), L_(Tensor::zeros({cfg.D, cfg.D}, true)) {}

void MGP::init() {
    mu_.fill(0.0f);
    for (int i = 0; i < cfg_.D; ++i) {
        for (int j = 0; j < cfg_.D; ++j) {
            L_.at(i * cfg_.D + j) = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void MGP::forward(const float* h_in, float* h_out) const {
    std::vector<float> centered(static_cast<size_t>(cfg_.D));
    std::vector<float> whitened(static_cast<size_t>(cfg_.D));
    for (int i = 0; i < cfg_.D; ++i) {
        centered[static_cast<size_t>(i)] = h_in[i] - mu_.at(i);
    }
    aurelis_cholesky_solve_lower(L_.data(), centered.data(), whitened.data(),
                                 static_cast<size_t>(cfg_.D));

    float norm = 0.0f;
    for (int i = 0; i < cfg_.D; ++i) {
        norm += whitened[static_cast<size_t>(i)] * whitened[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm + 1e-8f);
    const float scale = std::sqrt(static_cast<float>(cfg_.D)) / norm;
    for (int i = 0; i < cfg_.D; ++i) {
        h_out[i] = whitened[static_cast<size_t>(i)] * scale + mu_.at(i);
    }
}

void MGP::forward_sequence(const float* h_in, float* h_out, int n) const {
    for (int t = 0; t < n; ++t) {
        forward(h_in + t * cfg_.D, h_out + t * cfg_.D);
    }
}

void MGP::backward_step(const float* h_in, const float* grad_out, float* grad_in,
                       Tensor& grad_mu, Tensor& grad_L) const {
    const int D = cfg_.D;
    std::vector<float> centered(static_cast<size_t>(D));
    std::vector<float> whitened(static_cast<size_t>(D));
    for (int i = 0; i < D; ++i) {
        centered[static_cast<size_t>(i)] = h_in[i] - mu_.at(i);
    }
    /* Check for NaN in input */
    if (!std::isfinite(centered[0])) {
        std::memset(grad_in, 0, static_cast<size_t>(D) * sizeof(float));
        return;
    }
    if (aurelis_cholesky_solve_lower(L_.data(), centered.data(),
                                      whitened.data(),
                                      static_cast<size_t>(D)) != 0) {
        std::memset(grad_in, 0, static_cast<size_t>(D) * sizeof(float));
        return;
    }

    float norm = 0.0f;
    for (int i = 0; i < D; ++i) {
        norm += whitened[static_cast<size_t>(i)] * whitened[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm + 1e-8f);
    const float sqrt_D = std::sqrt(static_cast<float>(D));
    const float scale = sqrt_D / norm;

    std::vector<float> grad_z(static_cast<size_t>(D), 0.0f);
    float z_dot_gw = 0.0f;
    for (int i = 0; i < D; ++i) {
        float g = grad_out[i];
        if (!std::isfinite(g)) g = 0.0f;
        grad_z[static_cast<size_t>(i)] = g * scale;
        z_dot_gw += whitened[static_cast<size_t>(i)] * g;
    }
    float norm3 = norm * norm * norm;
    if (norm3 < 1e-12f) norm3 = 1e-12f;
    for (int i = 0; i < D; ++i) {
        grad_z[static_cast<size_t>(i)] -=
            (sqrt_D * z_dot_gw) / norm3 * whitened[static_cast<size_t>(i)];
    }

    std::vector<float> grad_x(static_cast<size_t>(D), 0.0f);
    for (int i = D - 1; i >= 0; --i) {
        float sum = grad_z[static_cast<size_t>(i)];
        for (int j = i + 1; j < D; ++j) {
            sum -= L_.at(static_cast<int64_t>(j * D + i)) *
                   grad_x[static_cast<size_t>(j)];
        }
        float diag = L_.at(static_cast<int64_t>(i * D + i));
        grad_x[static_cast<size_t>(i)] =
            (std::fabs(diag) > 1e-12f) ? sum / diag : 0.0f;
    }

    for (int i = 0; i < D; ++i) {
        grad_in[i] = grad_x[static_cast<size_t>(i)];
        grad_mu.at(i) += (std::isfinite(grad_out[i]) ? grad_out[i] : 0.0f) -
                         grad_x[static_cast<size_t>(i)];
    }

    /* dL_{ij} = - (L^{-T} grad_z)_i * (L^{-1} x)_j = -grad_x[i] * whitened[j] */
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j <= i; ++j) {
            grad_L.at(static_cast<int64_t>(i * D + j)) -=
                grad_x[static_cast<size_t>(i)] *
                whitened[static_cast<size_t>(j)];
        }
    }
}

}  // namespace aurelis::lens
