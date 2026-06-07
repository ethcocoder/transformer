#include "aurelis/aureum/cq.hpp"
#include "aurelis/lens/activations.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::aureum {

CQ::CQ(const AureumConfig& cfg)
    : cfg_(cfg),
      W_kappa_(aurelis::StatePartition::from_total_dim(cfg.cfg.D).De, 1) {}

void CQ::init() {
    W_kappa_.init_xavier();
}

float CQ::forward_step(const float* e_t) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    std::vector<int64_t> shape_e = {static_cast<int64_t>(part.De)};
    aurelis::Tensor e_tensor = aurelis::Tensor::from_data(shape_e, e_t, false);
    aurelis::Tensor kappa_out = W_kappa_.forward(e_tensor);

    // Sigmoid to get kappa in [0, 1]
    float s = 1.0f / (1.0f + std::exp(-kappa_out.at(0)));
    return s;
}

void CQ::forward_sequence(const float* e, int n, float* kappa) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    for (int t = 0; t < n; ++t) {
        kappa[t] = forward_step(e + t*part.De);
    }
}

void CQ::backward_step(const float* e_t, float grad_kappa,
                      float* grad_e,
                      aurelis::Tensor& grad_W_kappa, aurelis::Tensor& grad_b_kappa) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);

    // First compute kappa forward again to get pre-sigmoid value
    std::vector<int64_t> shape_e = {static_cast<int64_t>(part.De)};
    aurelis::Tensor e_tensor = aurelis::Tensor::from_data(shape_e, e_t, false);
    aurelis::Tensor kappa_out = W_kappa_.forward(e_tensor);
    float pre_sig = kappa_out.at(0);
    float s = 1.0f / (1.0f + std::exp(-pre_sig));
    float d_sigma = s * (1.0f - s);

    std::vector<int64_t> shape_gk = {1};
    std::vector<float> gk_data = {grad_kappa * d_sigma};
    aurelis::Tensor gk = aurelis::Tensor::from_data(shape_gk, gk_data.data(), false);

    std::vector<int64_t> shape_gh = {static_cast<int64_t>(part.De)};
    aurelis::Tensor gh = aurelis::Tensor::zeros(shape_gh);
    W_kappa_.backward(gk, e_tensor, grad_W_kappa, grad_b_kappa, gh);

    std::memcpy(grad_e, gh.data(), static_cast<std::size_t>(gh.numel()) * sizeof(float));
}

} // namespace aurelis::aureum
