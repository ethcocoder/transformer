#include "aurelis/lens/loss.hpp"

#include <algorithm>
#include <cmath>

namespace aurelis::lens {

float cross_entropy_next_token(const float* logits, const int* targets, int n,
                               int vocab, int d_model_stride,
                               int vocab_stride, std::vector<float>& grad_logits) {
    (void)d_model_stride;
    grad_logits.assign(static_cast<size_t>(n * vocab_stride), 0.0f);
    float loss = 0.0f;
    const int steps = n - 1;
    if (steps <= 0) {
        return 0.0f;
    }

    for (int t = 0; t < steps; ++t) {
        const float* row = logits + t * vocab_stride;
        float maxv = row[0];
        for (int v = 1; v < vocab; ++v) {
            maxv = std::max(maxv, row[v]);
        }
        if (!std::isfinite(maxv)) maxv = 0.0f;

        float sum = 0.0f;
        for (int v = 0; v < vocab; ++v) {
            float e = std::exp(row[v] - maxv);
            if (!std::isfinite(e)) e = 0.0f;
            sum += e;
        }
        sum = std::max(sum, 1e-30f);
        const float logsum = maxv + std::log(sum);

        int y = targets[t + 1];
        if (y < 0) y = 0;
        if (y >= vocab) y = vocab - 1;

        float ce = logsum - row[y];
        if (!std::isfinite(ce)) ce = 0.0f;
        loss += ce;

        for (int v = 0; v < vocab; ++v) {
            const float p = std::exp(row[v] - maxv) / sum;
            grad_logits[static_cast<size_t>(t * vocab_stride + v)] =
                std::isfinite(p) ? p : 0.0f;
        }
        grad_logits[static_cast<size_t>(t * vocab_stride + y)] -= 1.0f;
    }
    return loss / static_cast<float>(steps);
}

float stability_penalty(float max_forget, float eps_target) {
    const float viol = std::max(0.0f, max_forget - eps_target);
    return viol * viol;
}

}  // namespace aurelis::lens
