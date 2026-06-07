#include "aurelis/lens/ietcf.hpp"

#include "aurelis/orthogonal.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace aurelis::lens {

namespace {

/* R = (I - S)(I + S)^{-1}, S = A - A^T skew-symmetric. Stable vs Gram-Schmidt. */
void make_orthogonal_cayley(float* R, int d, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> S(static_cast<size_t>(d * d), 0.0f);
    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            const float v = dist(rng);
            S[static_cast<size_t>(i * d + j)] = v;
            S[static_cast<size_t>(j * d + i)] = -v;
        }
    }

    std::vector<float> M(static_cast<size_t>(d * d));
    std::vector<float> B(static_cast<size_t>(d * d));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const float s = S[static_cast<size_t>(i * d + j)];
            M[static_cast<size_t>(i * d + j)] = (i == j ? 1.0f : 0.0f) + s;
            B[static_cast<size_t>(i * d + j)] = (i == j ? 1.0f : 0.0f) - s;
        }
    }

    /* Solve M * R = B for R (Gauss-Jordan, d <= 512). */
    std::vector<float> Aaug(static_cast<size_t>(d * 2 * d));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            Aaug[static_cast<size_t>(i * 2 * d + j)] =
                M[static_cast<size_t>(i * d + j)];
            Aaug[static_cast<size_t>(i * 2 * d + d + j)] =
                B[static_cast<size_t>(i * d + j)];
        }
    }

    for (int col = 0; col < d; ++col) {
        int pivot = col;
        float maxv = std::fabs(Aaug[static_cast<size_t>(col * 2 * d + col)]);
        for (int r = col + 1; r < d; ++r) {
            const float v =
                std::fabs(Aaug[static_cast<size_t>(r * 2 * d + col)]);
            if (v > maxv) {
                maxv = v;
                pivot = r;
            }
        }
        if (maxv < 1e-12f) {
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    R[i * d + j] = (i == j) ? 1.0f : 0.0f;
                }
            }
            return;
        }
        if (pivot != col) {
            for (int j = 0; j < 2 * d; ++j) {
                std::swap(Aaug[static_cast<size_t>(col * 2 * d + j)],
                          Aaug[static_cast<size_t>(pivot * 2 * d + j)]);
            }
        }
        const float div =
            Aaug[static_cast<size_t>(col * 2 * d + col)];
        for (int j = 0; j < 2 * d; ++j) {
            Aaug[static_cast<size_t>(col * 2 * d + j)] /= div;
        }
        for (int r = 0; r < d; ++r) {
            if (r == col) {
                continue;
            }
            const float factor =
                Aaug[static_cast<size_t>(r * 2 * d + col)];
            for (int j = 0; j < 2 * d; ++j) {
                Aaug[static_cast<size_t>(r * 2 * d + j)] -=
                    factor * Aaug[static_cast<size_t>(col * 2 * d + j)];
            }
        }
    }

    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            R[i * d + j] = Aaug[static_cast<size_t>(i * 2 * d + d + j)];
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
    make_orthogonal_cayley(R_.data(), cfg_.d_tau, rng);
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
