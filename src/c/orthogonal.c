#include "aurelis/orthogonal.h"

void aurelis_orthogonal_apply(const float* R, const float* x, float* y,
                              size_t d) {
    for (size_t i = 0; i < d; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            sum += R[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}
