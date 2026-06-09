#ifndef AURELIS_MATMUL_H
#define AURELIS_MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C = alpha * A * B + beta * C, row-major. A:[M,K] B:[K,N] C:[M,N] */
void aurelis_matmul_f32(float alpha, const float* A, const float* B, float beta,
                        float* C, size_t M, size_t K, size_t N);

/* grad_A = grad_C @ B^T, grad_B = A^T @ grad_C */
void aurelis_matmul_backward_f32(const float* A, const float* B,
                                   const float* grad_C, float* grad_A,
                                   float* grad_B, size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif

#endif /* AURELIS_MATMUL_H */
