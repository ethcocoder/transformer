#pragma once

#include <cstddef>

namespace aurelis {

float sigmoid(float x);

/* out_i <- (1 - eps_k) * sigmoid(raw_i). */
void clamp_forget_gate(const float* raw, float* out, int D,
                       const float* eps_per_scale, const int* scale_index,
                       int num_scales);

}  // namespace aurelis
