#include "aurelis/convergence/efe.hpp"
#include <algorithm>
#include <vector>

namespace aurelis::convergence {

EFE::EFE(const EfeConfig& cfg)
    : cfg_(cfg),
      fce_mlp(256, cfg.D_bus),  // Dummy input size for now
      w_csp_world_(cfg.D_bus, 64),
      w_csp_motor_(cfg.D_bus, 64),
      w_csp_plan_(cfg.D_bus, 64) {}

void EFE::init() {
    fce_mlp.init_xavier();
    w_csp_world_.init_xavier();
    w_csp_motor_.init_xavier();
    w_csp_plan_.init_xavier();
}

EpistemicFrame EFE::assemble(const Tensor& c, const Tensor& e,
                              const Tensor& d, const Tensor& alpha,
                              float kappa, const Tensor& sigma, float H_reason) {
    EpistemicFrame f;
    f.c = c;
    f.e = e;
    f.d = d;
    f.alpha = alpha;
    f.kappa = kappa;
    f.sigma = sigma;
    f.H_reason = H_reason;
    return f;
}

Tensor EFE::encode_frame(const EpistemicFrame& f) {
    Tensor summary = Tensor::zeros({cfg_.D_bus});

    float content_sum = 0.0f;
    for (int64_t i = 0; i < f.c.numel(); ++i) content_sum += f.c.at(i);
    for (int64_t i = 0; i < f.e.numel(); ++i) content_sum += 0.5f * f.e.at(i);
    for (int64_t i = 0; i < f.d.numel(); ++i) content_sum += 0.25f * f.d.at(i);
    for (int64_t i = 0; i < f.alpha.numel(); ++i) content_sum += 0.1f * f.alpha.at(i);
    for (int64_t i = 0; i < f.sigma.numel(); ++i) content_sum += 0.05f * f.sigma.at(i);

    const float bias = 0.05f * (f.kappa + f.H_reason + 1.0f);
    for (int64_t i = 0; i < cfg_.D_bus; ++i) {
        summary.at(i) = bias + (content_sum / static_cast<float>(std::max<int64_t>(1, f.c.numel() + f.e.numel() + f.d.numel() + f.alpha.numel() + f.sigma.numel()))) + 0.01f * i;
    }

    return summary;
}

std::vector<Tensor> EFE::get_adapter_outputs(const Tensor& f_enc) {
    std::vector<Tensor> outputs;
    outputs.push_back(w_csp_world_.forward(f_enc));
    outputs.push_back(w_csp_motor_.forward(f_enc));
    outputs.push_back(w_csp_plan_.forward(f_enc));
    return outputs;
}

}  // namespace aurelis::convergence
