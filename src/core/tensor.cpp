#include "aurelis/tensor.hpp"

#include "aurelis/matmul.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace aurelis {

namespace {

void* aligned_alloc64(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 64);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

struct AlignedDeleter {
    void operator()(float* p) const { aligned_free(p); }
};

int64_t product(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

}  // namespace

Tensor::Tensor(std::vector<int64_t> shape, bool requires_grad)
    : shape_(std::move(shape)), requires_grad_(requires_grad) {
    const int64_t n = numel();
    float* raw = static_cast<float*>(aligned_alloc64(static_cast<size_t>(n) *
                                                     sizeof(float)));
    if (!raw) {
        throw std::bad_alloc();
    }
    data_ = std::shared_ptr<float>(raw, AlignedDeleter{});
    std::memset(data_.get(), 0, static_cast<size_t>(n) * sizeof(float));
}

Tensor Tensor::zeros(std::vector<int64_t> shape, bool requires_grad) {
    return Tensor(std::move(shape), requires_grad);
}

Tensor Tensor::from_data(std::vector<int64_t> shape, const float* data,
                         bool requires_grad) {
    Tensor t(std::move(shape), requires_grad);
    std::memcpy(t.data(), data, static_cast<size_t>(t.numel()) * sizeof(float));
    return t;
}

int64_t Tensor::numel() const { return product(shape_); }

int64_t Tensor::stride(int dim) const {
    int64_t s = 1;
    for (int i = ndim() - 1; i > dim; --i) {
        s *= shape_[i];
    }
    return s;
}

float& Tensor::at(int64_t i) { return data()[i]; }
float Tensor::at(int64_t i) const { return data()[i]; }

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    if (product(new_shape) != numel()) {
        throw std::invalid_argument("view: element count mismatch");
    }
    Tensor v = *this;
    v.shape_ = std::move(new_shape);
    return v;
}

void Tensor::fill(float value) {
    for (int64_t i = 0; i < numel(); ++i) {
        at(i) = value;
    }
}

Tensor Tensor::clone() const {
    return from_data(shape_, data(), requires_grad_);
}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.numel() != b.numel() || a.numel() != out.numel()) {
        throw std::invalid_argument("add: shape mismatch");
    }
    for (int64_t i = 0; i < a.numel(); ++i) {
        out.at(i) = a.at(i) + b.at(i);
    }
}

void mul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.numel() != b.numel() || a.numel() != out.numel()) {
        throw std::invalid_argument("mul: shape mismatch");
    }
    for (int64_t i = 0; i < a.numel(); ++i) {
        out.at(i) = a.at(i) * b.at(i);
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        throw std::invalid_argument("matmul: tensors must be 2D");
    }
    const size_t M = static_cast<size_t>(A.shape()[0]);
    const size_t K = static_cast<size_t>(A.shape()[1]);
    const size_t N = static_cast<size_t>(B.shape()[1]);
    if (B.shape()[0] != static_cast<int64_t>(K) ||
        C.shape()[0] != static_cast<int64_t>(M) ||
        C.shape()[1] != static_cast<int64_t>(N)) {
        throw std::invalid_argument("matmul: incompatible shapes");
    }
    aurelis_matmul_f32(1.0f, A.data(), B.data(), 0.0f, C.data(), M, K, N);
}

}  // namespace aurelis
