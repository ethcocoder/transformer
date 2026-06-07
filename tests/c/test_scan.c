#include "aurelis/scan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int feq(float a, float b, float tol) { return fabsf(a - b) <= tol; }

static int test_assoc(void) {
    if (aurelis_scan_op_assoc_test(1000, 1e-5f) != 0) {
        fprintf(stderr, "associativity test failed\n");
        return 1;
    }
    return 0;
}

static int test_forward_vs_sequential(size_t n, size_t D) {
    float* a = (float*)malloc(n * D * sizeof(float));
    float* b = (float*)malloc(n * D * sizeof(float));
    float* h0 = (float*)malloc(D * sizeof(float));
    float* out_scan = (float*)malloc(n * D * sizeof(float));
    float* out_seq = (float*)malloc(n * D * sizeof(float));

    for (size_t i = 0; i < n * D; ++i) {
        a[i] = 0.1f + 0.8f * ((float)rand() / RAND_MAX);
        b[i] = -0.5f + 1.0f * ((float)rand() / RAND_MAX);
    }
    for (size_t d = 0; d < D; ++d) {
        h0[d] = (float)rand() / RAND_MAX;
    }

    if (aurelis_scan_forward(a, b, h0, out_scan, n, D) != 0 ||
        aurelis_scan_forward_sequential(a, b, h0, out_seq, n, D) != 0) {
        free(a);
        free(b);
        free(h0);
        free(out_scan);
        free(out_seq);
        return 1;
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < n * D; ++i) {
        float err = fabsf(out_scan[i] - out_seq[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    free(a);
    free(b);
    free(h0);
    free(out_scan);
    free(out_seq);

    if (max_err > 1e-4f) {
        fprintf(stderr, "forward vs sequential max err %f (n=%zu D=%zu)\n",
                max_err, n, D);
        return 1;
    }
    return 0;
}

static int test_backward(size_t n, size_t D) {
    float* a = (float*)malloc(n * D * sizeof(float));
    float* b = (float*)malloc(n * D * sizeof(float));
    float* h0 = (float*)malloc(D * sizeof(float));
    float* h = (float*)malloc(n * D * sizeof(float));
    float* grad_out = (float*)malloc(n * D * sizeof(float));
    float* grad_a = (float*)malloc(n * D * sizeof(float));
    float* grad_b = (float*)malloc(n * D * sizeof(float));
    float* grad_h0 = (float*)malloc(D * sizeof(float));

    for (size_t i = 0; i < n * D; ++i) {
        a[i] = 0.3f + 0.4f * ((float)rand() / RAND_MAX);
        b[i] = -0.2f + 0.4f * ((float)rand() / RAND_MAX);
        grad_out[i] = (float)rand() / RAND_MAX;
    }
    for (size_t d = 0; d < D; ++d) {
        h0[d] = 0.5f;
    }

    aurelis_scan_forward_sequential(a, b, h0, h, n, D);
    aurelis_scan_backward(a, h, h0, grad_out, grad_a, grad_b, grad_h0, n, D);

    const float eps = 1e-3f;
    for (size_t t = 0; t < n; ++t) {
        for (size_t d = 0; d < D; ++d) {
            const size_t idx = t * D + d;

            float a_p = a[idx];
            a[idx] += eps;
            float loss_p = 0.0f;
            float hp[1024];
            aurelis_scan_forward_sequential(a, b, h0, hp, n, D);
            for (size_t i = 0; i < n * D; ++i) {
                loss_p += grad_out[i] * hp[i];
            }

            a[idx] = a_p - eps;
            float loss_m = 0.0f;
            float hm[1024];
            aurelis_scan_forward_sequential(a, b, h0, hm, n, D);
            for (size_t i = 0; i < n * D; ++i) {
                loss_m += grad_out[i] * hm[i];
            }
            a[idx] = a_p;

            float num = (loss_p - loss_m) / (2.0f * eps);
            if (!feq(num, grad_a[idx], 5e-2f)) {
                fprintf(stderr, "grad_a mismatch at %zu: num=%f ana=%f\n", idx,
                        num, grad_a[idx]);
                return 1;
            }
        }
    }

    free(a);
    free(b);
    free(h0);
    free(h);
    free(grad_out);
    free(grad_a);
    free(grad_b);
    free(grad_h0);
    return 0;
}

int main(void) {
    srand(42);
    if (test_assoc() != 0) {
        return 1;
    }
    const size_t sizes[] = {1, 7, 128, 1024};
    for (int i = 0; i < 4; ++i) {
        if (test_forward_vs_sequential(sizes[i], 8) != 0) {
            return 1;
        }
    }
    if (test_backward(16, 4) != 0) {
        return 1;
    }
    printf("test_scan_c: all passed\n");
    return 0;
}
