#ifndef AURELIS_LINALG_H
#define AURELIS_LINALG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Solve L x = b for lower-triangular L (row-major n x n). */
int aurelis_cholesky_solve_lower(const float* L, const float* b, float* x,
                                 size_t n);

/* Invert lower-triangular L in place into L_inv (row-major). */
int aurelis_tri_lower_inv(const float* L, float* L_inv, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* AURELIS_LINALG_H */
