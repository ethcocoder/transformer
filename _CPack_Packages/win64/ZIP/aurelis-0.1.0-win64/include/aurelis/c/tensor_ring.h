#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Tensor Ring decomposition structure
typedef struct {
    int order;       // Order of the tensor (number of modes)
    int rank;        // Tensor Ring rank (bond dimension)
    int* dims;       // Dimensions of each mode [order]
    float** cores;   // Tensor Ring cores: cores[i] is shape [rank, dims[i], rank]
} TensorRing;

// Initialize a Tensor Ring decomposition
void aurelis_tensor_ring_init(TensorRing* tr, int order, const int* dims, int rank);

// Free Tensor Ring memory
void aurelis_tensor_ring_free(TensorRing* tr);

// Perform Tensor Ring matvec without full materialization (CTR)
void aurelis_tensor_ring_matvec(const TensorRing* tr, const float* x, float* y);

// Reconstruct full tensor from Tensor Ring cores
void aurelis_tensor_ring_reconstruct(const TensorRing* tr, float* out);

#ifdef __cplusplus
}
#endif
