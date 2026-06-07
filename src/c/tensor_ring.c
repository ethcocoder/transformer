#include "aurelis/c/tensor_ring.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Initialize a Tensor Ring decomposition
void aurelis_tensor_ring_init(TensorRing* tr, int order, const int* dims, int rank) {
    tr->order = order;
    tr->rank = rank;
    tr->dims = (int*)malloc(order * sizeof(int));
    memcpy(tr->dims, dims, order * sizeof(int));
    tr->cores = (float**)malloc(order * sizeof(float*));

    // Initialize each core: [rank, dim, rank]
    for (int i = 0; i < order; ++i) {
        int core_size = rank * dims[i] * rank;
        tr->cores[i] = (float*)malloc(core_size * sizeof(float));
        // Initialize to random orthogonal-like for stability
        for (int j = 0; j < core_size; ++j) {
            tr->cores[i][j] = ((float)rand() / RAND_MAX) * 0.1f;
        }
    }
}

// Free Tensor Ring memory
void aurelis_tensor_ring_free(TensorRing* tr) {
    for (int i = 0; i < tr->order; ++i) {
        free(tr->cores[i]);
    }
    free(tr->cores);
    free(tr->dims);
    tr->order = 0;
    tr->rank = 0;
}

// Perform Tensor Ring matvec without full materialization (CTR: Contractive Tensor Ring)
void aurelis_tensor_ring_matvec(const TensorRing* tr, const float* x, float* y) {
    int rank = tr->rank;
    int n = 1;
    for (int i = 0; i < tr->order; ++i) {
        n *= tr->dims[i];
    }

    // First, reshape x to match the tensor dimensions
    // For simplicity, let's assume order=2 (matrix case) for now
    if (tr->order != 2) {
        // TODO: Handle higher orders
        return;
    }

    int dim0 = tr->dims[0];
    int dim1 = tr->dims[1];

    // Contract cores sequentially
    float* temp1 = (float*)malloc(rank * dim1 * sizeof(float));
    memset(temp1, 0, rank * dim1 * sizeof(float));

    // Contract core0 with x (dim0)
    for (int r0 = 0; r0 < rank; ++r0) {
        for (int d0 = 0; d0 < dim0; ++d0) {
            for (int r1 = 0; r1 < rank; ++r1) {
                float core0_val = tr->cores[0][r0 * dim0 * rank + d0 * rank + r1];
                temp1[r1 * dim1 + d0] += core0_val * x[d0 * dim1 + 0]; // TODO: Fix indexing for x
            }
        }
    }

    // Contract with core1 and produce y
    memset(y, 0, dim0 * dim1 * sizeof(float));
    for (int d0 = 0; d0 < dim0; ++d0) {
        for (int r0 = 0; r0 < rank; ++r0) {
            for (int d1 = 0; d1 < dim1; ++d1) {
                for (int r1 = 0; r1 < rank; ++r1) {
                    float core1_val = tr->cores[1][r0 * dim1 * rank + d1 * rank + r1];
                    y[d0 * dim1 + d1] += temp1[r0 * dim1 + d0] * core1_val;
                }
            }
        }
    }

    free(temp1);
}

// Reconstruct full tensor from Tensor Ring cores
void aurelis_tensor_ring_reconstruct(const TensorRing* tr, float* out) {
    // TODO: Implement full reconstruction
    (void)tr;
    (void)out;
}
