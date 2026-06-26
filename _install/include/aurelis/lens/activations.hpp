#pragma once

#include <cstddef>
#include <vector>

namespace aurelis::lens {

float silu(float x);
float dsilu(float x);

void silu_forward(const float* in, float* out, std::size_t n);
void silu_backward(const float* in, const float* grad_out, float* grad_in,
                   std::size_t n);

void sigmoid_forward(const float* in, float* out, std::size_t n);
void sigmoid_backward(const float* in, const float* grad_out, float* grad_in,
                      std::size_t n);

}  // namespace aurelis::lens
