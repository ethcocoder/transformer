#include "aurelis/tesserae/trfc.hpp"

#include <cstring>

namespace aurelis::tesserae {

TRFC::TRFC(const TRFCConfig& cfg) : cfg_(cfg) {
    tr_.order = 0;
    tr_.rank = 0;
    tr_.dims = nullptr;
    tr_.cores = nullptr;
}

TRFC::~TRFC() {
    if (tr_.order > 0) {
        aurelis_tensor_ring_free(&tr_);
    }
}

void TRFC::decompose_matrix(const Tensor& W) {
    original_shape_ = W.shape();

    // Reshape matrix to 2D tensor for order=2 TR
    int dims[2] = {static_cast<int>(original_shape_[0]), static_cast<int>(original_shape_[1])};

    // Initialize Tensor Ring
    aurelis_tensor_ring_init(&tr_, 2, dims, cfg_.tr_rank);

    // TODO: Implement proper STS (Structured Tensor Selection) using SVD for each unfolding
    // For now, leave cores as initialized random values
}

Tensor TRFC::matvec(const Tensor& x) const {
    if (tr_.order != 2) {
        return Tensor::zeros(x.shape());
    }

    Tensor y = Tensor::zeros({original_shape_[0]});
    aurelis_tensor_ring_matvec(&tr_, x.data(), y.data());
    return y;
}

Tensor TRFC::reconstruct() const {
    Tensor W = Tensor::zeros(original_shape_);
    aurelis_tensor_ring_reconstruct(&tr_, W.data());
    return W;
}

} // namespace aurelis::tesserae
