#include "aurelis/spectral.hpp"

#include <cmath>

namespace aurelis {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void clamp_forget_gate(const float* raw, float* out, int D,
                       const float* eps_per_scale, const int* scale_index,
                       int num_scales) {
    (void)num_scales;
    for (int i = 0; i < D; ++i) {
        const float eps = eps_per_scale[scale_index[i]];
        out[i] = (1.0f - eps) * sigmoid(raw[i]);
    }
}

}  // namespace aurelis
