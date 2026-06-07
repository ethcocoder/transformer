#ifndef AURELIS_SCAN_H
#define AURELIS_SCAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Affine semiring: (a2,b2) op (a1,b1) = (a2*a1, a2*b1+b2) */
void aurelis_scan_op(float a2, float b2, float a1, float b1,
                     float* out_a, float* out_b);

/* Inclusive scan per dimension: h[t,d] = prod(a[0..t,d])*h0[d] + sum(...). */
int aurelis_scan_forward(const float* a, const float* b, const float* h0,
                         float* out, size_t n, size_t D);

/* Sequential reference (same result as scan_forward). */
int aurelis_scan_forward_sequential(const float* a, const float* b,
                                    const float* h0, float* out,
                                    size_t n, size_t D);

/* Reverse-mode adjoint for sequential recurrence. */
int aurelis_scan_backward(const float* a, const float* h, const float* h0,
                          const float* grad_out, float* grad_a, float* grad_b,
                          float* grad_h0, size_t n, size_t D);

/* Returns 0 on success, non-zero if associativity check fails. */
int aurelis_scan_op_assoc_test(int trials, float tol);

#ifdef __cplusplus
}
#endif

#endif /* AURELIS_SCAN_H */
