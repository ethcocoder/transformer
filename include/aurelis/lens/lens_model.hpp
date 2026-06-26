#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/ietcf.hpp"
#include "aurelis/lens/lens_layer.hpp"
#include "aurelis/lens/osh.hpp"

#include <vector>

namespace aurelis::lens {

struct LensForwardResult {
    float loss_ce = 0.0f;
    float loss_aux = 0.0f;
    float loss_stab = 0.0f;
    float loss_total = 0.0f;
    std::vector<float> logits;
};

class LensModel {
 public:
    explicit LensModel(LensConfig cfg);

    void init();

    LensForwardResult forward(const int* tokens, int n);
    LensForwardResult train_step(const int* tokens, int n);

    LensConfig& config() { return cfg_; }
    IETCF& ietcf() { return ietcf_; }
    OSH& osh() { return osh_; }
    std::vector<LensLayer>& layers() { return layers_; }

 private:
    LensConfig cfg_;
    IETCF ietcf_;
    std::vector<LensLayer> layers_;
    OSH osh_;

    std::vector<float> x_embed_;
    std::vector<float> gamma_;
    std::vector<float> x_stream_;
    std::vector<float> final_c_;
    std::vector<LayerCache> caches_;
    std::vector<std::vector<float>> layer_inputs_;

    void sgd_step(Tensor& param, const Tensor& grad, float lr);
};

}  // namespace aurelis::lens
