#ifndef AURELIS_RECURRENCE_H
#define AURELIS_RECURRENCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* h[t] = a[t]*h[t-1] + b[t], h[-1] = h0. Layout: [n, D] row-major. */
int aurelis_recurrence_diag_forward(const float* a, const float* b,
                                    const float* h0, float* h,
                                    size_t n, size_t D);

#ifdef __cplusplus
}
#endif

#endif /* AURELIS_RECURRENCE_H */
