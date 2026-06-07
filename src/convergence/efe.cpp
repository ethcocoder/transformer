#include "aurelis/convergence/efe.hpp"
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
    // Concatenate relevant parts of f and apply fce MLP
    // For now, use dummy tensor
    return Tensor::zeros({cfg_.D_bus});
}

std::vector<Tensor> EFE::get_adapter_outputs(const Tensor& f_enc) {
    std::vector<Tensor> outputs;
    outputs.push_back(w_csp_world_.forward(f_enc));
    outputs.push_back(w_csp_motor_.forward(f_enc));
    outputs.push_back(w_csp_plan_.forward(f_enc));
    return outputs;
}

}  // namespace aurelis::convergence
