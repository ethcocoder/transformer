#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include "aurelis/errors.hpp"

namespace aurelis {

class Tensor {
 public:
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, bool requires_grad = false);

    static Tensor zeros(std::vector<int64_t> shape, bool requires_grad = false);
    static Tensor from_data(std::vector<int64_t> shape, const float* data,
                            bool requires_grad = false);

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

    const std::vector<int64_t>& shape() const { return shape_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int64_t numel() const;
    int64_t stride(int dim) const;

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v) { requires_grad_ = v; }

    float& at(int64_t i);
    float at(int64_t i) const;

    Tensor view(std::vector<int64_t> new_shape) const;
    Tensor clone() const;
    void fill(float value);

    // Checkpointing
    bool save(const std::string& filepath) const;
    static Tensor load(const std::string& filepath);

 private:
    std::shared_ptr<float> data_;
    std::vector<int64_t> shape_;
    bool requires_grad_ = false;
};

void add(const Tensor& a, const Tensor& b, Tensor& out);
void mul(const Tensor& a, const Tensor& b, Tensor& out);
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

}  // namespace aurelis
