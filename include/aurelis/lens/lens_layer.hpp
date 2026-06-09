#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/csc.hpp"
#include "aurelis/lens/fwse.hpp"
#include "aurelis/lens/mgp.hpp"
#include "aurelis/lens/spi.hpp"

#include <vector>

namespace aurelis::lens {

struct LayerCache {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> h_raw;
    std::vector<float> h_csc;
    std::vector<float> h_mgp;
    std::vector<float> c;
    std::vector<float> e;
    std::vector<float> r;
    std::vector<float> m;
    float aux_loss = 0.0f;
};

class LensLayer {
 public:
    explicit LensLayer(const LensConfig& cfg);

    void init();

    /* x_stream[n*d_model], gamma[n*d_model] -> x_out[n*d_model], fills cache */
    void forward(const float* x_stream, const float* gamma, int n,
                 float* x_out, LayerCache& cache) const;

    void backward(const float* x_stream, const float* gamma, const float* grad_c,
                  int n, const LayerCache& cache,
                  float* grad_x_stream, float* grad_gamma,
                  Tensor& grad_gate_W, Tensor& grad_gate_b,
                  Tensor& grad_ctrl_W, Tensor& grad_ctrl_b,
                  Tensor& grad_Wa, Tensor& grad_ba, Tensor& grad_Wb,
                  Tensor& grad_bb, Tensor& grad_Winj, Tensor& grad_binj,
                  Tensor& grad_M, Tensor& grad_c_bias,
                  Tensor& grad_mu, Tensor& grad_L,
                  Tensor& grad_We, Tensor& grad_be, Tensor& grad_Wr,
                  Tensor& grad_br, Tensor& grad_Wm,
                  Tensor& grad_bm) const;

    FWSE& fwse() { return fwse_; }
    CSC& csc() { return csc_; }
    MGP& mgp() { return mgp_; }
    SPI& spi() { return spi_; }

 private:
    LensConfig cfg_;
    FWSE fwse_;
    CSC csc_;
    MGP mgp_;
    SPI spi_;
};

}  // namespace aurelis::lens
