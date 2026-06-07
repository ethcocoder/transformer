#include "aurelis/linalg.h"

#include <string.h>

int aurelis_cholesky_solve_lower(const float* L, const float* b, float* x,
                                 size_t n) {
    if (!L || !b || !x || n == 0) {
        return -1;
    }

    float tmp[512];
    if (n > 512) {
        return -1;
    }

    /* Forward substitution: L y = b */
    for (size_t i = 0; i < n; ++i) {
        float sum = b[i];
        for (size_t j = 0; j < i; ++j) {
            sum -= L[i * n + j] * tmp[j];
        }
        const float diag = L[i * n + i];
        if (diag == 0.0f) {
            return -1;
        }
        tmp[i] = sum / diag;
    }

    memcpy(x, tmp, n * sizeof(float));
    return 0;
}

int aurelis_tri_lower_inv(const float* L, float* L_inv, size_t n) {
    if (!L || !L_inv || n == 0) {
        return -1;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            L_inv[i * n + j] = 0.0f;
        }
    }

    for (size_t col = 0; col < n; ++col) {
        const float diag = L[col * n + col];
        if (diag == 0.0f) {
            return -1;
        }
        L_inv[col * n + col] = 1.0f / diag;
        for (size_t i = col + 1; i < n; ++i) {
            float sum = 0.0f;
            for (size_t j = col; j < i; ++j) {
                sum += L[i * n + j] * L_inv[j * n + col];
            }
            L_inv[i * n + col] = -sum / L[i * n + i];
        }
    }

    return 0;
}
