#include "aurelis/lens/lens_model.hpp"

#include "aurelis/lens/loss.hpp"

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
    float max_forget = 0.0f;
    float aux_sum = 0.0f;

    std::vector<float> x_next(static_cast<size_t>(n * cfg_.d_model));
    for (int l = 0; l < cfg_.num_layers; ++l) {
        layers_[static_cast<size_t>(l)].forward(
            x_stream_.data(), gamma_.data(), n, x_next.data(),
            caches_[static_cast<size_t>(l)]);
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

  return result;
}

}  // namespace aurelis::lens
