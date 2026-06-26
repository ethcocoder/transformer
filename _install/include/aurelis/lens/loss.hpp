#pragma once

#include <vector>

namespace aurelis::lens {

float cross_entropy_next_token(const float* logits, const int* targets, int n,
                               int vocab, int d_model_stride,
                               int vocab_stride, std::vector<float>& grad_logits);

float stability_penalty(float max_forget, float eps_target = 0.99f);

}  // namespace aurelis::lens
