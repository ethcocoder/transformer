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
    // Step 1: Compute forward pass values again
    std::vector<float> centered(static_cast<size_t>(D));
    std::vector<float> whitened(static_cast<size_t>(D));
    for (int i = 0; i < D; ++i) {
        centered[static_cast<size_t>(i)] = h_in[i] - mu_.at(i);
    }
    aurelis_cholesky_solve_lower(L_.data(), centered.data(), whitened.data(),
                                 static_cast<size_t>(D));
    float norm = 0.0f;
    for (int i = 0; i < D; ++i) {
        norm += whitened[static_cast<size_t>(i)] * whitened[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm + 1e-8f);
    const float sqrt_D = std::sqrt(static_cast<float>(D));
    const float scale = sqrt_D / norm;

    // Step 2: Compute gradient for h_out to whitened (dw -> dz)
    std::vector<float> grad_z(static_cast<size_t>(D), 0.0f);
    float z_dot_gw = 0.0f;
    for (int i = 0; i < D; ++i) {
        grad_z[static_cast<size_t>(i)] = grad_out[i] * scale;
        z_dot_gw += whitened[static_cast<size_t>(i)] * grad_out[i];
    }
    for (int i = 0; i < D; ++i) {
        grad_z[static_cast<size_t>(i)] -= (sqrt_D * z_dot_gw) / (norm * norm * norm) * whitened[static_cast<size_t>(i)];
    }

    // Step 3: Compute gradient for whitened to centered (dz -> dx) using L^{-T} (since z = L^{-1} x => dz/dx = L^{-T})
    std::vector<float> grad_x(static_cast<size_t>(D), 0.0f);
    // First, solve L^T grad_x = grad_z (forward substitution for lower triangular L^T, which is upper triangular)
    for (int i = D - 1; i >= 0; --i) {
        float sum = grad_z[static_cast<size_t>(i)];
        for (int j = i + 1; j < D; ++j) {
            sum -= L_.at(j * D + i) * grad_x[static_cast<size_t>(j)];
        }
        grad_x[static_cast<size_t>(i)] = sum / L_.at(i * D + i);
    }

    // Step 4: Compute grad_in and grad_mu
    for (int i = 0; i < D; ++i) {
        grad_in[i] = grad_x[static_cast<size_t>(i)];
        grad_mu.at(i) += grad_out[i] - grad_x[static_cast<size_t>(i)];
    }

    // Step 5: Compute gradient for L (dL)
    // First, compute the outer product of grad_z and whitened (wait, actually, dL is a bit trickier; let's use the formula for derivative of L^{-1} x)
    // The derivative of z = L^{-1} x w.r.t L is -L^{-1} x (L^{-1})^T, multiplied by grad_z
    std::vector<float> L_inv(static_cast<size_t>(D * D), 0.0f);
    aurelis_tri_lower_inv(L_.data(), L_inv.data(), static_cast<size_t>(D));

    // Compute B = L^{-T} grad_z (which we already have as grad_x) and A = L^{-1} centered (which is whitened)
    // Then, dL_{ij} = -sum_{k} A_k B_k if i >= j (since L is lower triangular)
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += whitened[static_cast<size_t>(k)] * grad_x[static_cast<size_t>(k)];
            }
            // Wait, actually, more accurately: d/dL_{ij} (L^{-1} x) = - (L^{-1})_i (L^{-T} x)_j = - (L^{-1} x)_i (L^{-T} x)_j? Wait no, let's think again
            // Let me get this right: if we have z = L^{-1} x, then dz/dL is -L^{-1} dL L^{-1} x
            // So the contribution to dL_{ij} is - (L^{-T} grad_z)_i (L^{-1} x)_j
            // Yes, that's the correct formula!
            grad_L.at(i * D + j) -= grad_x[static_cast<size_t>(i)] * whitened[static_cast<size_t>(j)];
        }
    }
}

}  // namespace aurelis::lens
