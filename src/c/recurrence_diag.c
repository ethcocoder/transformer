#include "aurelis/recurrence.h"

int aurelis_recurrence_diag_forward(const float* a, const float* b,
                                    const float* h0, float* h, size_t n,
                                    size_t D) {
    if (!a || !b || !h0 || !h || n == 0 || D == 0) {
        return -1;
    }

    for (size_t d = 0; d < D; ++d) {
        float prev = h0[d];
        for (size_t t = 0; t < n; ++t) {
            const size_t idx = t * D + d;
            prev = a[idx] * prev + b[idx];
            h[idx] = prev;
        }
    }
    return 0;
}
