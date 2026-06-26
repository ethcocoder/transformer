#include "aurelis/lens/lens_model.hpp"

#include "aurelis/lens/loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aurelis::lens {

LensModel::LensModel(LensConfig cfg)
    : cfg_(cfg), ietcf_(cfg), osh_(cfg) {
    layers_.reserve(static_cast<size_t>(cfg.num_layers));
    for (int i = 0; i < cfg.num_layers; ++i) {
        layers_.emplace_back(cfg);
    }
}

void LensModel::init() {
    ietcf_.init();
    for (auto& layer : layers_) {
        layer.init();
    }
    osh_.init();
}

LensForwardResult LensModel::forward(const int* tokens, int n) {
    LensForwardResult result;
    if (n < 2) {
        return result;
    }

    x_embed_.resize(static_cast<size_t>(n * cfg_.d_model));
    gamma_.resize(static_cast<size_t>(n * cfg_.d_model));
    ietcf_.forward(tokens, n, x_embed_.data(), gamma_.data());

    x_stream_ = x_embed_;
    caches_.resize(static_cast<size_t>(cfg_.num_layers));
    layer_inputs_.resize(static_cast<size_t>(cfg_.num_layers));
    float max_forget = 0.0f;
    float aux_sum = 0.0f;

    std::vector<float> x_next(static_cast<size_t>(n * cfg_.d_model));
    for (int l = 0; l < cfg_.num_layers; ++l) {
        layer_inputs_[static_cast<size_t>(l)] = x_stream_;
        layers_[static_cast<size_t>(l)].forward(
            x_stream_.data(), gamma_.data(), n, x_next.data(),
            caches_[static_cast<size_t>(l)]);
        /* Production NaN guard: reset if layer output contains NaN */
        bool nan_detected = false;
        for (size_t i = 0; i < x_next.size(); ++i) {
            if (!std::isfinite(x_next[i])) {
                nan_detected = true;
                break;
            }
        }
        if (nan_detected) {
            x_next = x_stream_;
        }
        x_stream_.swap(x_next);
        max_forget = std::max(
            max_forget,
            layers_[static_cast<size_t>(l)].fwse().max_forget(
                caches_[static_cast<size_t>(l)].a.data(), n));
        aux_sum += caches_[static_cast<size_t>(l)].aux_loss;
    }

    final_c_ = caches_.back().c;
    result.logits.resize(static_cast<size_t>(n * cfg_.vocab_size));
    osh_.forward_sequence(final_c_.data(), x_embed_.data(), n,
                          result.logits.data());

    std::vector<float> grad_logits;
    result.loss_ce =
        cross_entropy_next_token(result.logits.data(), tokens, n,
                               cfg_.vocab_size, cfg_.d_model, cfg_.vocab_size,
                               grad_logits);
    result.loss_aux = aux_sum / static_cast<float>(cfg_.num_layers);
    result.loss_stab =
        stability_penalty(max_forget) * cfg_.lambda_stab;
    result.loss_total =
        result.loss_ce + cfg_.lambda_aux * result.loss_aux + result.loss_stab;

    if (!std::isfinite(result.loss_total)) {
        result.loss_total = 10.0f;
        result.loss_ce = 10.0f;
        result.loss_aux = 0.0f;
        result.loss_stab = 0.0f;
    }
    return result;
}

void LensModel::sgd_step(Tensor& param, const Tensor& grad, float lr) {
    for (int64_t i = 0; i < param.numel(); ++i) {
        const float g = grad.at(i);
        if (std::isfinite(g)) {
            param.at(i) -= lr * g;
        }
    }
}

LensForwardResult LensModel::train_step(const int* tokens, int n) {
    LensForwardResult result = forward(tokens, n);
    if (n < 2) {
        return result;
    }

    std::vector<float> grad_logits;
    cross_entropy_next_token(result.logits.data(), tokens, n, cfg_.vocab_size,
                             cfg_.d_model, cfg_.vocab_size, grad_logits);

    std::vector<float> grad_c(static_cast<size_t>(n * cfg_.d_model), 0.0f);
    std::vector<float> grad_x_embed(static_cast<size_t>(n * cfg_.d_model),
                                    0.0f);

    Tensor grad_Wout = Tensor::zeros(osh_.out_proj().weight().shape());
    Tensor grad_bout = Tensor::zeros(osh_.out_proj().bias().shape());
    Tensor grad_Wskip = Tensor::zeros(osh_.skip_proj().weight().shape());
    Tensor grad_bskip = Tensor::zeros(osh_.skip_proj().bias().shape());

    const int steps = n - 1;
    for (int t = 0; t < steps; ++t) {
        osh_.backward_step(grad_logits.data() + t * cfg_.vocab_size,
                           final_c_.data() + t * cfg_.d_model,
                           x_embed_.data() + t * cfg_.d_model,
                           grad_c.data() + t * cfg_.d_model,
                           grad_x_embed.data() + t * cfg_.d_model, grad_Wout,
                           grad_bout, grad_Wskip, grad_bskip);
    }

    sgd_step(osh_.out_proj().weight(), grad_Wout, cfg_.lr);
    sgd_step(osh_.out_proj().bias(), grad_bout, cfg_.lr);
    sgd_step(osh_.skip_proj().weight(), grad_Wskip, cfg_.lr);
    sgd_step(osh_.skip_proj().bias(), grad_bskip, cfg_.lr);

    Tensor grad_E = Tensor::zeros(ietcf_.embedding().shape());
    for (int t = 0; t < steps; ++t) {
        const int tok = tokens[t];
        for (int d = 0; d < cfg_.d_model; ++d) {
            grad_E.at(tok * cfg_.d_model + d) +=
                grad_x_embed[static_cast<size_t>(t * cfg_.d_model + d)];
        }
    }
    sgd_step(ietcf_.embedding(), grad_E, cfg_.lr);

    std::vector<float> grad_next = grad_c;
    for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
        std::vector<float> grad_x_stream(static_cast<size_t>(n * cfg_.d_model), 0.0f);
        std::vector<float> grad_gamma(static_cast<size_t>(n * cfg_.d_model), 0.0f);

        Tensor grad_gate_W = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().gate().weight().shape());
        Tensor grad_gate_b = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().gate().bias().shape());
        Tensor grad_ctrl_W = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().ctrl().weight().shape());
        Tensor grad_ctrl_b = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().ctrl().bias().shape());
        Tensor grad_Wa = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().W_a().weight().shape());
        Tensor grad_ba = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().W_a().bias().shape());
        Tensor grad_Wb = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().W_b().weight().shape());
        Tensor grad_bb = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().W_b().bias().shape());
        Tensor grad_Winj = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().W_inj().weight().shape());
        Tensor grad_binj = Tensor::zeros(layers_[static_cast<size_t>(l)].fwse().W_inj().bias().shape());
        Tensor grad_M = Tensor::zeros(layers_[static_cast<size_t>(l)].csc().mix_matrix().shape());
        Tensor grad_c_bias = Tensor::zeros(layers_[static_cast<size_t>(l)].csc().mix_bias().shape());
        Tensor grad_mu = Tensor::zeros(layers_[static_cast<size_t>(l)].mgp().mu().shape());
        Tensor grad_L = Tensor::zeros(layers_[static_cast<size_t>(l)].mgp().L().shape());
        Tensor grad_We = Tensor::zeros(layers_[static_cast<size_t>(l)].spi().W_e().weight().shape());
        Tensor grad_be = Tensor::zeros(layers_[static_cast<size_t>(l)].spi().W_e().bias().shape());
        Tensor grad_Wr = Tensor::zeros(layers_[static_cast<size_t>(l)].spi().W_r().weight().shape());
        Tensor grad_br = Tensor::zeros(layers_[static_cast<size_t>(l)].spi().W_r().bias().shape());
        Tensor grad_Wm = Tensor::zeros(layers_[static_cast<size_t>(l)].spi().W_m().weight().shape());
        Tensor grad_bm = Tensor::zeros(layers_[static_cast<size_t>(l)].spi().W_m().bias().shape());

        layers_[static_cast<size_t>(l)].backward(
            layer_inputs_[static_cast<size_t>(l)].data(), gamma_.data(),
            grad_next.data(), n, caches_[static_cast<size_t>(l)],
            grad_x_stream.data(), grad_gamma.data(), grad_gate_W, grad_gate_b,
            grad_ctrl_W, grad_ctrl_b, grad_Wa, grad_ba, grad_Wb, grad_bb,
            grad_Winj, grad_binj, grad_M, grad_c_bias, grad_mu, grad_L,
            grad_We, grad_be, grad_Wr, grad_br, grad_Wm, grad_bm);

        sgd_step(layers_[static_cast<size_t>(l)].fwse().gate().weight(), grad_gate_W, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().gate().bias(), grad_gate_b, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().ctrl().weight(), grad_ctrl_W, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().ctrl().bias(), grad_ctrl_b, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().W_a().weight(), grad_Wa, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().W_a().bias(), grad_ba, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().W_b().weight(), grad_Wb, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().W_b().bias(), grad_bb, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().W_inj().weight(), grad_Winj, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].fwse().W_inj().bias(), grad_binj, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].csc().mix_matrix(), grad_M, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].csc().mix_bias(), grad_c_bias, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].mgp().mu(), grad_mu, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].mgp().L(), grad_L, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].spi().W_e().weight(), grad_We, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].spi().W_e().bias(), grad_be, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].spi().W_r().weight(), grad_Wr, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].spi().W_r().bias(), grad_br, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].spi().W_m().weight(), grad_Wm, cfg_.lr);
        sgd_step(layers_[static_cast<size_t>(l)].spi().W_m().bias(), grad_bm, cfg_.lr);

        grad_next = grad_x_stream;
    }

    return result;
}

}  // namespace aurelis::lens
