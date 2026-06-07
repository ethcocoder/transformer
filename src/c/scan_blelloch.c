#include "aurelis/scan.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    float alpha;
    float beta;
} ScanPair;

void aurelis_scan_op(float a2, float b2, float a1, float b1, float* out_a,
                     float* out_b) {
    *out_a = a2 * a1;
    *out_b = a2 * b1 + b2;
}

static ScanPair scan_combine(ScanPair right, ScanPair left) {
    ScanPair out;
    aurelis_scan_op(right.alpha, right.beta, left.alpha, left.beta, &out.alpha,
                    &out.beta);
    return out;
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

#if defined(__GNUC__)
__attribute__((unused))
#endif
static int blelloch_exclusive(const ScanPair* in, ScanPair* exclusive, size_t n) {
    const size_t m = next_pow2(n);
    ScanPair* buf = (ScanPair*)calloc(m, sizeof(ScanPair));
    if (!buf) {
        return -1;
    }

    memcpy(buf, in, n * sizeof(ScanPair));
    for (size_t i = n; i < m; ++i) {
        buf[i].alpha = 1.0f;
        buf[i].beta = 0.0f;
    }

    /* Up-sweep */
    for (size_t d = 0; d < (size_t)log2((double)m); ++d) {
        size_t stride = (size_t)1 << (d + 1);
        for (size_t i = stride - 1; i < m; i += stride) {
            size_t left = i - (stride >> 1);
            buf[i] = scan_combine(buf[i], buf[left]);
        }
    }

  buf[m - 1].alpha = 1.0f;
  buf[m - 1].beta = 0.0f;

    /* Down-sweep */
    for (int d = (int)log2((double)m) - 1; d >= 0; --d) {
        size_t stride = (size_t)1 << (d + 1);
        for (size_t i = stride - 1; i < m; i += stride) {
            size_t left = i - (stride >> 1);
            ScanPair temp = buf[left];
            buf[left] = buf[i];
            buf[i] = scan_combine(buf[i], temp);
        }
    }

    memcpy(exclusive, buf, n * sizeof(ScanPair));
    free(buf);
    return 0;
}

int aurelis_scan_forward_sequential(const float* a, const float* b,
                                    const float* h0, float* out, size_t n,
                                    size_t D) {
    if (!a || !b || !h0 || !out || n == 0 || D == 0) {
        return -1;
    }

    for (size_t d = 0; d < D; ++d) {
        float prev = h0[d];
        for (size_t t = 0; t < n; ++t) {
            const size_t idx = t * D + d;
            prev = a[idx] * prev + b[idx];
            out[idx] = prev;
        }
    }
    return 0;
}

int aurelis_scan_forward(const float* a, const float* b, const float* h0,
                         float* out, size_t n, size_t D) {
    if (!a || !b || !h0 || !out || n == 0 || D == 0) {
        return -1;
    }

    /* Inclusive prefix scan: O(n) work per dimension, exact for affine recurrence.
     * Blelloch (blelloch_exclusive) is retained for future parallel/GPU paths. */
    for (size_t d = 0; d < D; ++d) {
        ScanPair prefix = {1.0f, 0.0f};
        for (size_t t = 0; t < n; ++t) {
            const size_t idx = t * D + d;
            ScanPair u = {a[idx], b[idx]};
            prefix = scan_combine(u, prefix);
            out[idx] = prefix.alpha * h0[d] + prefix.beta;
        }
    }

    return 0;
}

int aurelis_scan_backward(const float* a, const float* h, const float* h0,
                          const float* grad_out, float* grad_a, float* grad_b,
                          float* grad_h0, size_t n, size_t D) {
    if (!a || !h || !h0 || !grad_out || !grad_a || !grad_b || !grad_h0 ||
        n == 0 || D == 0) {
        return -1;
    }

    memset(grad_a, 0, n * D * sizeof(float));
    memset(grad_b, 0, n * D * sizeof(float));
    memset(grad_h0, 0, D * sizeof(float));

    for (size_t d = 0; d < D; ++d) {
        float grad_h_prev = 0.0f;

        for (int t = (int)n - 1; t >= 0; --t) {
            const size_t idx = (size_t)t * D + d;
            const float g = grad_out[idx] + grad_h_prev;
            const float h_prev =
                (t == 0) ? h0[d] : h[(size_t)(t - 1) * D + d];
            grad_a[idx] = g * h_prev;
            grad_b[idx] = g;
            grad_h_prev = g * a[idx];
        }

        grad_h0[d] = grad_h_prev;
    }

    return 0;
}

int aurelis_scan_op_assoc_test(int trials, float tol) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }
    for (int i = 0; i < trials; ++i) {
        ScanPair u1 = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
        ScanPair u2 = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
        ScanPair u3 = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};

        ScanPair left = scan_combine(scan_combine(u3, u2), u1);
        ScanPair right = scan_combine(u3, scan_combine(u2, u1));

        if (fabsf(left.alpha - right.alpha) > tol ||
            fabsf(left.beta - right.beta) > tol) {
            return 1;
        }
    }
    return 0;
}
