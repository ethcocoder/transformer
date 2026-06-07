#include "aurelis/matmul.h"

#ifdef AURELIS_USE_BLAS
#include <cblas.h>
#endif

static void matmul_naive(float alpha, const float* A, const float* B, float beta,
                         float* C, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void aurelis_matmul_f32(float alpha, const float* A, const float* B, float beta,
                        float* C, size_t M, size_t K, size_t N) {
#ifdef AURELIS_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N,
                (int)K, alpha, A, (int)K, B, (int)N, beta, C, (int)N);
#else
    matmul_naive(alpha, A, B, beta, C, M, K, N);
#endif
}

void aurelis_matmul_backward_f32(const float* A, const float* B,
                                 const float* grad_C, float* grad_A,
                                 float* grad_B, size_t M, size_t K, size_t N) {
#ifdef AURELIS_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)M, (int)K, (int)N,
                1.0f, grad_C, (int)N, B, (int)N, 0.0f, grad_A, (int)K);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)K, (int)N, (int)M,
                1.0f, A, (int)K, grad_C, (int)N, 0.0f, grad_B, (int)N);
#else
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float sum = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                sum += grad_C[i * N + n] * B[k * N + n];
            }
            grad_A[i * K + k] = sum;
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t m = 0; m < M; ++m) {
                sum += A[m * K + k] * grad_C[m * N + n];
            }
            grad_B[k * N + n] = sum;
        }
    }
#endif
}
