#include "aurelis/aureum/ese.hpp"
#include "aurelis/lens/activations.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::aureum {

ESE::ESE(const AureumConfig& cfg)
    : cfg_(cfg),
      W_ec_(cfg.cfg.d_model, aurelis::StatePartition::from_total_dim(cfg.cfg.D).De),
      W_sigma_(aurelis::StatePartition::from_total_dim(cfg.cfg.D).De, aurelis::StatePartition::from_total_dim(cfg.cfg.D).De) {}

void ESE::init() {
    W_ec_.init_xavier();
    W_sigma_.init_xavier();
}

float ESE::forward_step(const float* c_t, const float* /*m_prev*/, const float* gate, float* e_t) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);

    // Step 1: Compute injection from c_t
    std::vector<int64_t> shape_c = {static_cast<int64_t>(cfg_.cfg.d_model)};
    aurelis::Tensor c_tensor = aurelis::Tensor::from_data(shape_c, c_t, false);
    aurelis::Tensor e_inj = W_ec_.forward(c_tensor);

    // Step 2: Apply SiLU to e_inj
    std::vector<float> e_silu(static_cast<size_t>(part.De));
    aurelis::lens::silu_forward(e_inj.data(), e_silu.data(), static_cast<size_t>(e_inj.numel()));

    // Step 3: Apply MEG gate
    for (int i = 0; i < part.De; ++i) {
        e_t[i] = e_silu[static_cast<size_t>(i)] * gate[i];
    }

    // Step 4: UVH: compute sigma = softplus(W_sigma @ e_t + b_sigma)
    aurelis::Tensor e_tensor = aurelis::Tensor::from_data({static_cast<int64_t>(part.De)}, e_t, false);
    aurelis::Tensor sigma_pre = W_sigma_.forward(e_tensor);
    float sigma = 0.0f;
    // Take mean of sigma_pre, apply softplus
    for (int i = 0; i < part.De; ++i) {
        sigma += std::log1pf(std::exp(sigma_pre.data()[i])); // softplus
    }
    sigma /= static_cast<float>(part.De);

    return sigma;
}

void ESE::forward_sequence(const float* c, const float* m, const float* gates, int n, float* e) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    for (int t = 0; t < n; ++t) {
        const float* m_prev = (t > 0) ? (m + (t-1)*part.Dm) : nullptr;
        const float* gate_t = gates + t*part.De;
        forward_step(c + t*cfg_.cfg.d_model, m_prev, gate_t, e + t*part.De);
    }
}

void ESE::backward_step(const float* c_t, const float* m_prev, const float* gate, const float* grad_e,
                       float* grad_c, float* grad_m,
                       aurelis::Tensor& grad_W_ec, aurelis::Tensor& grad_b_ec,
                       aurelis::Tensor& grad_W_sigma, aurelis::Tensor& grad_b_sigma) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);

    // Step 1: Compute forward pass values again for backprop
    std::vector<int64_t> shape_c = {static_cast<int64_t>(cfg_.cfg.d_model)};
    aurelis::Tensor c_tensor = aurelis::Tensor::from_data(shape_c, c_t, false);
    aurelis::Tensor e_inj = W_ec_.forward(c_tensor);

    std::vector<float> e_silu(static_cast<size_t>(part.De));
    aurelis::lens::silu_forward(e_inj.data(), e_silu.data(), static_cast<size_t>(e_inj.numel()));

    // Step 2: Backward through gate application
    std::vector<float> grad_e_silu(static_cast<size_t>(part.De));
    for (int i = 0; i < part.De; ++i) {
        grad_e_silu[static_cast<size_t>(i)] = grad_e[i] * gate[i];
    }

    // Step 3: Backward through SiLU
    std::vector<float> silu_grad(static_cast<size_t>(part.De));
    for (int i = 0; i < part.De; ++i) {
        float x = e_inj.data()[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        silu_grad[static_cast<size_t>(i)] = grad_e_silu[static_cast<size_t>(i)] * (sig * (1.0f + x * (1.0f - sig)));
    }

    // Step 4: Backward through W_ec_
    std::vector<int64_t> shape_e = {static_cast<int64_t>(part.De)};
    aurelis::Tensor grad_e_tensor = aurelis::Tensor::from_data(shape_e, silu_grad.data(), false);
    std::vector<int64_t> shape_gc = {static_cast<int64_t>(cfg_.cfg.d_model)};
    aurelis::Tensor grad_c_tensor = aurelis::Tensor::zeros(shape_gc);
    W_ec_.backward(grad_e_tensor, c_tensor, grad_W_ec, grad_b_ec, grad_c_tensor);
    std::memcpy(grad_c, grad_c_tensor.data(), static_cast<size_t>(grad_c_tensor.numel()) * sizeof(float));

    // Step 5: Backward through UVH (W_sigma_)
    // For now, we'll assume sigma isn't in the loss, but we'll still compute the gradients for completeness
    // First, compute e_t again
    std::vector<float> e_t(static_cast<size_t>(part.De));
    for (int i = 0; i < part.De; ++i) {
        e_t[static_cast<size_t>(i)] = e_silu[static_cast<size_t>(i)] * gate[i];
    }
    aurelis::Tensor e_tensor = aurelis::Tensor::from_data(shape_e, e_t.data(), false);
    aurelis::Tensor sigma_pre = W_sigma_.forward(e_tensor);
    // Backward pass for W_sigma_ (even though sigma isn't in the loss yet, we'll implement it for when it is!)
    // For now, we'll just compute the gradient as if sigma was in the loss with a dummy grad
    std::vector<float> dummy_grad_sigma_pre(static_cast<size_t>(part.De), 0.0f);
    for (int i = 0; i < part.De; ++i) {
        float x = sigma_pre.data()[i];
        // derivative of softplus is sigmoid(x), but we don't need it right now since dummy grad is zero
        (void)x; // to avoid unused variable warning
        dummy_grad_sigma_pre[static_cast<size_t>(i)] = 0.0f; // no gradient for now
    }
    aurelis::Tensor grad_sigma_pre_tensor = aurelis::Tensor::from_data(shape_e, dummy_grad_sigma_pre.data(), false);
    aurelis::Tensor grad_e_uvh = aurelis::Tensor::zeros(shape_e);
    W_sigma_.backward(grad_sigma_pre_tensor, e_tensor, grad_W_sigma, grad_b_sigma, grad_e_uvh);

    // Step 6: Set grad_m to zero for now
    if (m_prev != nullptr) {
        std::memset(grad_m, 0, static_cast<size_t>(part.Dm) * sizeof(float));
    }
}

} // namespace aurelis::aureum
