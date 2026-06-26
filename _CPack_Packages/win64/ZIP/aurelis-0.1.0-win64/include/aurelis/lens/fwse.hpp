#pragma once

#include "aurelis/lens/config.hpp"
#include "aurelis/lens/linear.hpp"
#include "aurelis/lens/mssp.hpp"

namespace aurelis::lens {

class FWSE {
 public:
    explicit FWSE(const LensConfig& cfg);

    void init();

    /* Per timestep: x[d_model], gamma[d_model] -> a[D], b[D] */
    void forward_step(const float* x, const float* gamma, float* a,
                      float* b) const;

    /* Sequence: x[n*d_model], gamma[n*d_model] -> a[n*D], b[n*D] */
    void forward_sequence(const float* x, const float* gamma, int n, float* a,
                          float* b) const;

    float max_forget(const float* a, int n) const;

 private:
    LensConfig cfg_;
    MsspLayout mssp_;
    Linear gate_;
    Linear ctrl_;
    Linear W_a_;
    Linear W_b_;
    Linear W_inj_;
};

}  // namespace aurelis::lens
