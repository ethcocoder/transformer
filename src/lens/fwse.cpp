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
