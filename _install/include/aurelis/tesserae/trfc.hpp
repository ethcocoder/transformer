#pragma once

#include <aurelis/tensor.hpp>
#include <aurelis/c/tensor_ring.h>
#include <vector>

namespace aurelis::tesserae {

struct TRFCConfig {
    int tr_rank;         // Tensor Ring rank
    float target_error;  // Target reconstruction error for STS (Structured Tensor Selection)
};

class TRFC {
public:
    explicit TRFC(const TRFCConfig& cfg);

    ~TRFC();

    // Decompose a matrix into Tensor Ring (order=2 for matrices)
    void decompose_matrix(const Tensor& W);

    // Multiply with vector using Tensor Ring (no full materialization)
    Tensor matvec(const Tensor& x) const;

    // Reconstruct the full matrix
    Tensor reconstruct() const;

    // Get the Tensor Ring structure
    const TensorRing* tr() const { return &tr_; }

private:
    TRFCConfig cfg_;
    TensorRing tr_;
    std::vector<int64_t> original_shape_;
};

} // namespace aurelis::tesserae
