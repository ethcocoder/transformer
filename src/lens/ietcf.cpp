#include "aurelis/lens/ietcf.hpp"

#include "aurelis/orthogonal.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace aurelis::lens {

namespace {

/* Generate orthogonal matrix via QR decomposition of a random normal matrix.
 * Gram-Schmidt is numerically stable for moderate dimensions (d <= 512) and
 * avoids the singularity issues of the Cayley Gauss-Jordan approach. */
void make_orthogonal_qr(float* R, int d, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> A(static_cast<size_t>(d * d));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A[static_cast<size_t>(i * d + j)] = dist(rng);
        }
    }

    /* Modified Gram-Schmidt: A = QR, extract Q into R */
    std::vector<float> Q(static_cast<size_t>(d * d), 0.0f);
    for (int k = 0; k < d; ++k) {
        float norm_sq = 0.0f;
        for (int i = 0; i < d; ++i) {
            norm_sq += A[static_cast<size_t>(i * d + k)] *
                       A[static_cast<size_t>(i * d + k)];
        }
        float norm = std::sqrt(norm_sq + 1e-12f);
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < d; ++i) {
            Q[static_cast<size_t>(i * d + k)] =
                A[static_cast<size_t>(i * d + k)] * inv_norm;
        }
        for (int j = k + 1; j < d; ++j) {
            float dot = 0.0f;
            for (int i = 0; i < d; ++i) {
                dot += Q[static_cast<size_t>(i * d + k)] *
                       A[static_cast<size_t>(i * d + j)];
            }
            for (int i = 0; i < d; ++i) {
                A[static_cast<size_t>(i * d + j)] -=
                    dot * Q[static_cast<size_t>(i * d + k)];
            }
        }
    }

    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            R[static_cast<size_t>(i * d + j)] =
                Q[static_cast<size_t>(i * d + j)];
        }
    }
}

}  // namespace

IETCF::IETCF(const LensConfig& cfg)
    : cfg_(cfg),
      E_(Tensor::zeros({cfg.vocab_size, cfg.d_model}, true)),
      R_(Tensor::zeros({cfg.d_tau, cfg.d_tau}, true)),
      W_gamma_(cfg.d_tau, cfg.d_model),
      tau_(static_cast<size_t>(cfg.d_tau), 0.0f) {}

void IETCF::init() {
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (int64_t i = 0; i < E_.numel(); ++i) {
        E_.at(i) = dist(rng);
    }
    make_orthogonal_qr(R_.data(), cfg_.d_tau, rng);
    W_gamma_.init_xavier();
    for (int i = 0; i < cfg_.d_tau; ++i) {
        tau_[static_cast<size_t>(i)] = (i == 0) ? 1.0f : 0.0f;
    }
}

void IETCF::forward(const int* tokens, int n, float* x_embed, float* gamma) {
    for (int t = 0; t < n; ++t) {
        const int tok = tokens[t];
        if (tok < 0 || tok >= cfg_.vocab_size) {
            continue;
        }
        std::memcpy(x_embed + t * cfg_.d_model,
                    E_.data() + tok * cfg_.d_model,
                    static_cast<size_t>(cfg_.d_model) * sizeof(float));
    }

    std::vector<float> tau_new(static_cast<size_t>(cfg_.d_tau));
    for (int t = 0; t < n; ++t) {
        aurelis_orthogonal_apply(R_.data(), tau_.data(), tau_new.data(),
                                 static_cast<size_t>(cfg_.d_tau));
        tau_.swap(tau_new);

        Tensor tau_tensor =
            Tensor::from_data({cfg_.d_tau}, tau_.data(), false);
        Tensor g = W_gamma_.forward(tau_tensor);
        std::memcpy(gamma + t * cfg_.d_model, g.data(),
                    static_cast<size_t>(cfg_.d_model) * sizeof(float));
    }
}

}  // namespace aurelis::lens
