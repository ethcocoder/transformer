#pragma once

namespace aurelis::lens {

struct LensConfig {
    int vocab_size = 64;
    int D = 64;           /* hidden state dimension */
    int d_model = 32;     /* embedding / content / residual stream (= D_c) */
    int d_tau = 32;       /* TCG oscillator dimension */
    int d_ff = 256;       /* FWSE expansion (= 4*D typical) */
    int num_layers = 2;
    int num_scales = 4;
    float eps_scales[4] = {0.9f, 0.5f, 0.1f, 0.01f};
    float lambda_stab = 0.01f;
    float lambda_aux = 0.01f;
    float lr = 1e-2f;

    int Dc() const { return d_model; }
    int De() const { return static_cast<int>(0.2f * D); }
    int Dr() const { return static_cast<int>(0.2f * D); }
    int Dm() const { return D - Dc() - De() - Dr(); }
};

}  // namespace aurelis::lens
