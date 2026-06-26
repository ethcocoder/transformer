#pragma once

#include "aurelis/convergence/common.hpp"
#include "aurelis/tensor.hpp"
#include "aurelis/lens/linear.hpp"

namespace aurelis::convergence {

struct EfeConfig {
    int D_bus;
};

class EFE {
public:
    explicit EFE(const EfeConfig& cfg);

    void init();

    EpistemicFrame assemble(const Tensor& c, const Tensor& e,
                            const Tensor& d, const Tensor& alpha,
                            float kappa, const Tensor& sigma, float H_reason);

    Tensor encode_frame(const EpistemicFrame& f);

    std::vector<Tensor> get_adapter_outputs(const Tensor& f_enc);
// Get components
    aurelis::lens::Linear& fce() { return fce_mlp; }
    aurelis::lens::Linear& w_csp_world() { return w_csp_world_; }
    aurelis::lens::Linear& w_csp_motor() { return w_csp_motor_; }
    aurelis::lens::Linear& w_csp_plan() { return w_csp_plan_; }

private:
    EfeConfig cfg_;
    aurelis::lens::Linear fce_mlp;
    aurelis::lens::Linear w_csp_world_;
    aurelis::lens::Linear w_csp_motor_;
    aurelis::lens::Linear w_csp_plan_;
};

}  // namespace aurelis::convergence
