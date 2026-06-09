#ifndef AURELIS_ORTHOGONAL_H
#define AURELIS_ORTHOGONAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y = R @ x, R is d x d row-major orthogonal. */
void aurelis_orthogonal_apply(const float* R, const float* x, float* y,
                              size_t d);

#ifdef __cplusplus
}
#endif

#endif /* AURELIS_ORTHOGONAL_H */
