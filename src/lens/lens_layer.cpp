#include "aurelis/lens/lens_layer.hpp"

#include <cstring>
#include <vector>

namespace aurelis::lens {

LensLayer::LensLayer(const LensConfig& cfg)
    : cfg_(cfg), fwse_(cfg), csc_(cfg), mgp_(cfg), spi_(cfg) {}

void LensLayer::init() {
    fwse_.init();
    csc_.init();
    mgp_.init();
    spi_.init();
}

void LensLayer::forward(const float* x_stream, const float* gamma, int n,
                        float* x_out, LayerCache& cache) const {
    const int D = cfg_.D;
    const int Dc = cfg_.Dc();
    const int De = cfg_.De();
    const int Dr = cfg_.Dr();
    const int Dm = cfg_.Dm();

    cache.a.resize(static_cast<size_t>(n * D));
    cache.b.resize(static_cast<size_t>(n * D));
    cache.h_raw.resize(static_cast<size_t>(n * D));
    cache.h_csc.resize(static_cast<size_t>(n * D));
    cache.h_mgp.resize(static_cast<size_t>(n * D));
    cache.c.resize(static_cast<size_t>(n * Dc));
    cache.e.resize(static_cast<size_t>(n * De));
    cache.r.resize(static_cast<size_t>(n * Dr));
    cache.m.resize(static_cast<size_t>(n * Dm));

    fwse_.forward_sequence(x_stream, gamma, n, cache.a.data(), cache.b.data());
    csc_.forward(cache.a.data(), cache.b.data(), n, cache.h_csc.data(),
                 cache.h_raw.data());
    mgp_.forward_sequence(cache.h_csc.data(), cache.h_mgp.data(), n);
    spi_.forward_sequence(cache.h_mgp.data(), cache.c.data(), cache.e.data(),
                          cache.r.data(), cache.m.data(), n, cache.aux_loss);

    for (int t = 0; t < n; ++t) {
        for (int d = 0; d < Dc; ++d) {
            x_out[t * Dc + d] =
                cache.c[t * Dc + d] + x_stream[t * Dc + d];
        }
    }
}

void LensLayer::backward(const float* x_stream, const float* gamma,
                         const float* grad_c, int n, const LayerCache& cache,
                         float* grad_x_stream, float* grad_gamma,
                         Tensor& grad_gate_W, Tensor& grad_gate_b,
                         Tensor& grad_ctrl_W, Tensor& grad_ctrl_b,
                         Tensor& grad_Wa, Tensor& grad_ba, Tensor& grad_Wb,
                         Tensor& grad_bb, Tensor& grad_Winj,
                         Tensor& grad_binj, Tensor& grad_M,
                         Tensor& grad_c_bias, Tensor& grad_mu,
                         Tensor& grad_L, Tensor& grad_We, Tensor& grad_be,
                         Tensor& grad_Wr, Tensor& grad_br, Tensor& grad_Wm,
                         Tensor& grad_bm) const {
    const int D = cfg_.D;
    const int De = cfg_.De();
    const int Dr = cfg_.Dr();
    const int Dm = cfg_.Dm();

    std::vector<float> grad_h_mgp(static_cast<size_t>(n * D), 0.0f);
    std::vector<float> grad_h_csc(static_cast<size_t>(n * D), 0.0f);
    std::vector<float> grad_h_raw(static_cast<size_t>(n * D), 0.0f);
    std::vector<float> grad_a(static_cast<size_t>(n * D), 0.0f);
    std::vector<float> grad_b(static_cast<size_t>(n * D), 0.0f);
    std::vector<float> grad_e_reg(static_cast<size_t>(n * De), 0.0f);
    std::vector<float> grad_r_reg(static_cast<size_t>(n * Dr), 0.0f);
    std::vector<float> grad_m_reg(static_cast<size_t>(n * Dm), 0.0f);

    const float aux_scale = cfg_.lambda_aux / static_cast<float>(cfg_.num_layers);
    for (int t = 0; t < n; ++t) {
        for (int i = 0; i < De; ++i) {
            grad_e_reg[static_cast<size_t>(t * De + i)] =
                2.0f * cache.e[static_cast<size_t>(t * De + i)] * aux_scale /
                static_cast<float>(n);
        }
        for (int i = 0; i < Dr; ++i) {
            grad_r_reg[static_cast<size_t>(t * Dr + i)] =
                2.0f * cache.r[static_cast<size_t>(t * Dr + i)] * aux_scale /
                static_cast<float>(n);
        }
        for (int i = 0; i < Dm; ++i) {
            grad_m_reg[static_cast<size_t>(t * Dm + i)] =
                2.0f * cache.m[static_cast<size_t>(t * Dm + i)] * aux_scale /
                static_cast<float>(n);
        }
    }

    spi_.backward_step(cache.h_mgp.data(), grad_c, grad_e_reg.data(),
                       grad_r_reg.data(), grad_m_reg.data(),
                       grad_h_mgp.data(), grad_We, grad_be, grad_Wr,
                       grad_br, grad_Wm, grad_bm);

    for (int t = 0; t < n; ++t) {
        mgp_.backward_step(cache.h_csc.data() + t * D,
                           grad_h_mgp.data() + t * D,
                           grad_h_csc.data() + t * D, grad_mu, grad_L);
    }

    csc_.backward(cache.a.data(), cache.h_raw.data(), grad_h_csc.data(), n,
                  grad_a.data(), grad_b.data(), grad_h_raw.data(),
                  grad_M, grad_c_bias);

    fwse_.backward_step(x_stream, gamma, grad_a.data(), grad_b.data(), n,
                        grad_gate_W, grad_gate_b, grad_ctrl_W, grad_ctrl_b,
                        grad_Wa, grad_ba, grad_Wb, grad_bb, grad_Winj,
                        grad_binj, grad_x_stream, grad_gamma);
}

}  // namespace aurelis::lens
