#include "aurelis/lens/fwse.hpp"

#include "aurelis/lens/activations.hpp"
#include "aurelis/spectral.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::lens {

FWSE::FWSE(const LensConfig& cfg)
    : cfg_(cfg),
      mssp_(MsspLayout::build(cfg.D, cfg.num_scales)),
      gate_(cfg.d_model, cfg.d_ff),
      ctrl_(cfg.d_model, cfg.d_ff),
      W_a_(cfg.d_ff, cfg.D),
      W_b_(cfg.d_ff, cfg.D),
      W_inj_(cfg.d_ff, cfg.D) {}

void FWSE::init() {
    gate_.init_xavier();
    ctrl_.init_xavier();
    W_a_.init_xavier();
    W_b_.init_xavier();
    W_inj_.init_xavier();
}

void FWSE::forward_step(const float* x, const float* gamma, float* a,
                        float* b) const {
    std::vector<float> gate_pre(static_cast<size_t>(cfg_.d_ff));
    std::vector<float> ctrl_pre(static_cast<size_t>(cfg_.d_ff));
    linear_forward(gate_.weight().data(), gate_.bias().data(), x,
                   gate_pre.data(), cfg_.d_ff, cfg_.d_model);
    linear_forward(ctrl_.weight().data(), ctrl_.bias().data(), gamma,
                   ctrl_pre.data(), cfg_.d_ff, cfg_.d_model);

    std::vector<float> z(static_cast<size_t>(cfg_.d_ff));
    for (int i = 0; i < cfg_.d_ff; ++i) {
        z[static_cast<size_t>(i)] =
            silu(gate_pre[static_cast<size_t>(i)] +
                 ctrl_pre[static_cast<size_t>(i)]);
    }

    std::vector<float> a_tilde(static_cast<size_t>(cfg_.D));
    std::vector<float> b_sig(static_cast<size_t>(cfg_.D));
    std::vector<float> b_inj(static_cast<size_t>(cfg_.D));
    linear_forward(W_a_.weight().data(), W_a_.bias().data(), z.data(),
                   a_tilde.data(), cfg_.D, cfg_.d_ff);
    linear_forward(W_b_.weight().data(), W_b_.bias().data(), z.data(),
                   b_sig.data(), cfg_.D, cfg_.d_ff);
    linear_forward(W_inj_.weight().data(), W_inj_.bias().data(), z.data(),
                   b_inj.data(), cfg_.D, cfg_.d_ff);

    clamp_forget_gate(a_tilde.data(), a, cfg_.D, cfg_.eps_scales,
                      mssp_.scale_index.data(), cfg_.num_scales);

    silu_forward(b_inj.data(), b_inj.data(), static_cast<size_t>(cfg_.D));
    sigmoid_forward(b_sig.data(), b_sig.data(), static_cast<size_t>(cfg_.D));
    for (int i = 0; i < cfg_.D; ++i) {
        b[i] = b_sig[static_cast<size_t>(i)] * b_inj[static_cast<size_t>(i)];
    }
}

void FWSE::forward_sequence(const float* x, const float* gamma, int n,
                            float* a, float* b) const {
    for (int t = 0; t < n; ++t) {
        forward_step(x + t * cfg_.d_model, gamma + t * cfg_.d_model,
                     a + t * cfg_.D, b + t * cfg_.D);
    }
}

void FWSE::backward_step(const float* x, const float* gamma,
                         const float* grad_a, const float* grad_b, int n,
                         Tensor& grad_gate_W, Tensor& grad_gate_b,
                         Tensor& grad_ctrl_W, Tensor& grad_ctrl_b,
                         Tensor& grad_Wa, Tensor& grad_ba, Tensor& grad_Wb,
                         Tensor& grad_bb, Tensor& grad_Winj,
                         Tensor& grad_binj, float* grad_x,
                         float* grad_gamma) const {
    std::vector<float> gate_pre(static_cast<size_t>(cfg_.d_ff));
    std::vector<float> ctrl_pre(static_cast<size_t>(cfg_.d_ff));
    std::vector<float> z(static_cast<size_t>(cfg_.d_ff));
    std::vector<float> a_tilde(static_cast<size_t>(cfg_.D));
    std::vector<float> b_sig(static_cast<size_t>(cfg_.D));
    std::vector<float> b_inj(static_cast<size_t>(cfg_.D));
    std::vector<float> grad_z(static_cast<size_t>(cfg_.d_ff), 0.0f);
    std::vector<float> grad_pre(static_cast<size_t>(cfg_.d_ff), 0.0f);

    std::fill(grad_x, grad_x + static_cast<size_t>(n * cfg_.d_model), 0.0f);
    std::fill(grad_gamma, grad_gamma + static_cast<size_t>(n * cfg_.d_model), 0.0f);

    for (int t = 0; t < n; ++t) {
        linear_forward(gate_.weight().data(), gate_.bias().data(),
                       x + t * cfg_.d_model, gate_pre.data(), cfg_.d_ff,
                       cfg_.d_model);
        linear_forward(ctrl_.weight().data(), ctrl_.bias().data(),
                       gamma + t * cfg_.d_model, ctrl_pre.data(), cfg_.d_ff,
                       cfg_.d_model);

        for (int i = 0; i < cfg_.d_ff; ++i) {
            z[static_cast<size_t>(i)] =
                silu(gate_pre[static_cast<size_t>(i)] +
                     ctrl_pre[static_cast<size_t>(i)]);
        }

        linear_forward(W_a_.weight().data(), W_a_.bias().data(), z.data(),
                       a_tilde.data(), cfg_.D, cfg_.d_ff);
        linear_forward(W_b_.weight().data(), W_b_.bias().data(), z.data(),
                       b_sig.data(), cfg_.D, cfg_.d_ff);
        linear_forward(W_inj_.weight().data(), W_inj_.bias().data(), z.data(),
                       b_inj.data(), cfg_.D, cfg_.d_ff);

        std::vector<float> grad_a_raw(static_cast<size_t>(cfg_.D), 0.0f);
        std::vector<float> grad_b_sig(static_cast<size_t>(cfg_.D), 0.0f);
        std::vector<float> grad_b_inj(static_cast<size_t>(cfg_.D), 0.0f);
        std::vector<float> grad_z_a(static_cast<size_t>(cfg_.d_ff), 0.0f);
        std::vector<float> grad_z_b(static_cast<size_t>(cfg_.d_ff), 0.0f);
        std::vector<float> grad_z_inj(static_cast<size_t>(cfg_.d_ff), 0.0f);

        for (int i = 0; i < cfg_.D; ++i) {
            const float eps = cfg_.eps_scales[mssp_.scale_index[static_cast<size_t>(i)]];
            const float s = 1.0f / (1.0f + std::exp(-a_tilde[i]));
            const float ds = s * (1.0f - s);
            grad_a_raw[static_cast<size_t>(i)] = grad_a[t * cfg_.D + i] * (1.0f - eps) * ds;

            const float s_b = 1.0f / (1.0f + std::exp(-b_sig[i]));
            const float ds_b = s_b * (1.0f - s_b);
            const float ds_inj = dsilu(b_inj[i]);
            grad_b_sig[static_cast<size_t>(i)] = grad_b[t * cfg_.D + i] * silu(b_inj[i]) * ds_b;
            grad_b_inj[static_cast<size_t>(i)] = grad_b[t * cfg_.D + i] * s_b * ds_inj;
        }

        linear_backward(W_a_.weight().data(), z.data(), grad_a_raw.data(),
                        grad_Wa.data(), grad_ba.data(), grad_z_a.data(),
                        cfg_.D, cfg_.d_ff);
        linear_backward(W_b_.weight().data(), z.data(), grad_b_sig.data(),
                        grad_Wb.data(), grad_bb.data(), grad_z_b.data(),
                        cfg_.D, cfg_.d_ff);
        linear_backward(W_inj_.weight().data(), z.data(), grad_b_inj.data(),
                        grad_Winj.data(), grad_binj.data(), grad_z_inj.data(),
                        cfg_.D, cfg_.d_ff);

        for (int i = 0; i < cfg_.d_ff; ++i) {
            grad_z[i] += grad_z_a[static_cast<size_t>(i)] +
                         grad_z_b[static_cast<size_t>(i)] +
                         grad_z_inj[static_cast<size_t>(i)];
        }

        std::vector<float> grad_pre_local(static_cast<size_t>(cfg_.d_ff), 0.0f);
        for (int i = 0; i < cfg_.d_ff; ++i) {
            grad_pre_local[static_cast<size_t>(i)] = grad_z[i] * dsilu(gate_pre[i] + ctrl_pre[i]);
        }

        std::vector<float> grad_x_gate(static_cast<size_t>(cfg_.d_model), 0.0f);
        std::vector<float> grad_gamma_ctrl(static_cast<size_t>(cfg_.d_model), 0.0f);
        linear_backward(gate_.weight().data(), x + t * cfg_.d_model,
                        grad_pre_local.data(), grad_gate_W.data(),
                        grad_gate_b.data(), grad_x_gate.data(), cfg_.d_ff,
                        cfg_.d_model);
        linear_backward(ctrl_.weight().data(), gamma + t * cfg_.d_model,
                        grad_pre_local.data(), grad_ctrl_W.data(),
                        grad_ctrl_b.data(), grad_gamma_ctrl.data(), cfg_.d_ff,
                        cfg_.d_model);

        for (int i = 0; i < cfg_.d_model; ++i) {
            grad_x[t * cfg_.d_model + i] += grad_x_gate[static_cast<size_t>(i)];
            grad_gamma[t * cfg_.d_model + i] +=
                grad_gamma_ctrl[static_cast<size_t>(i)];
        }
    }
}

float FWSE::max_forget(const float* a, int n) const {
    float m = 0.0f;
    for (int t = 0; t < n; ++t) {
        for (int d = 0; d < cfg_.D; ++d) {
            m = std::max(m, std::fabs(a[t * cfg_.D + d]));
        }
    }
    return m;
}

}  // namespace aurelis::lens
