#include "aurelis/lens/spi.hpp"

#include <cmath>
#include <cstring>

namespace aurelis::lens {

SPI::SPI(const LensConfig& cfg)
    : cfg_(cfg),
      part_(StatePartition::from_total_dim(cfg.D)),
      W_e_(cfg.D, part_.De),
      W_r_(cfg.D, part_.Dr),
      W_m_(cfg.D, part_.Dm) {}

void SPI::init() {
    W_e_.init_xavier();
    W_r_.init_xavier();
    W_m_.init_xavier();
}

float SPI::forward_step(const float* h, float* c, float* e, float* r,
                        float* m) const {
    std::memcpy(c, h, static_cast<size_t>(cfg_.Dc()) * sizeof(float));

    Tensor h_t = Tensor::from_data({cfg_.D}, h, false);
    Tensor e_t = W_e_.forward(h_t);
    Tensor r_t = W_r_.forward(h_t);
    Tensor m_t = W_m_.forward(h_t);
    std::memcpy(e, e_t.data(), static_cast<size_t>(part_.De) * sizeof(float));
    std::memcpy(r, r_t.data(), static_cast<size_t>(part_.Dr) * sizeof(float));
    std::memcpy(m, m_t.data(), static_cast<size_t>(part_.Dm) * sizeof(float));

    float aux = 0.0f;
    for (int i = 0; i < part_.De; ++i) {
        aux += e[i] * e[i];
    }
    for (int i = 0; i < part_.Dr; ++i) {
        aux += r[i] * r[i];
    }
    for (int i = 0; i < part_.Dm; ++i) {
        aux += m[i] * m[i];
    }
    return aux;
}

void SPI::forward_sequence(const float* h, float* c, float* e, float* r,
                           float* m, int n, float& aux_loss) const {
    aux_loss = 0.0f;
    for (int t = 0; t < n; ++t) {
        aux_loss += forward_step(h + t * cfg_.D, c + t * cfg_.Dc(),
                                 e + t * part_.De, r + t * part_.Dr,
                                 m + t * part_.Dm);
    }
    aux_loss /= static_cast<float>(n);
}

void SPI::backward_step(const float* h, const float* grad_c, const float* grad_e,
                        const float* grad_r, const float* grad_m, float* grad_h,
                        Tensor& grad_We, Tensor& grad_be, Tensor& grad_Wr,
                        Tensor& grad_br, Tensor& grad_Wm,
                        Tensor& grad_bm) const {
    std::memcpy(grad_h, grad_c, static_cast<size_t>(cfg_.Dc()) * sizeof(float));
    for (int i = cfg_.Dc(); i < cfg_.D; ++i) {
        grad_h[i] = 0.0f;
    }

    Tensor h_t = Tensor::from_data({cfg_.D}, h, false);
    Tensor ge = Tensor::from_data({part_.De}, grad_e, false);
    Tensor gr = Tensor::from_data({part_.Dr}, grad_r, false);
    Tensor gm = Tensor::from_data({part_.Dm}, grad_m, false);
    Tensor gh = Tensor::zeros({cfg_.D});

    W_e_.backward(ge, h_t, grad_We, grad_be, gh);
    for (int i = 0; i < cfg_.D; ++i) {
        grad_h[i] += gh.at(i);
        gh.at(i) = 0.0f;
    }
    W_r_.backward(gr, h_t, grad_Wr, grad_br, gh);
    for (int i = 0; i < cfg_.D; ++i) {
        grad_h[i] += gh.at(i);
        gh.at(i) = 0.0f;
    }
    W_m_.backward(gm, h_t, grad_Wm, grad_bm, gh);
    for (int i = 0; i < cfg_.D; ++i) {
        grad_h[i] += gh.at(i);
    }
}

}  // namespace aurelis::lens
