#include "aurelis/lens/linear.hpp"

#include "aurelis/matmul.h"

#include <cmath>
#include <cstring>
#include <random>

namespace aurelis::lens {

Linear::Linear(int in_features, int out_features)
    : in_(in_features),
      out_(out_features),
      W_(Tensor::zeros({out_features, in_features}, true)),
      b_(Tensor::zeros({out_features}, true)) {}

void Linear::init_xavier() {
    std::mt19937 rng(42);
    const float scale =
        std::sqrt(2.0f / static_cast<float>(in_ + out_));
    std::normal_distribution<float> dist(0.0f, scale);
    for (int64_t i = 0; i < W_.numel(); ++i) {
        W_.at(i) = dist(rng);
    }
    b_.fill(0.0f);
}

void linear_forward(const float* W, const float* b, const float* x, float* y,
                    int out_f, int in_f) {
    aurelis_matmul_f32(1.0f, W, x, 0.0f, y, static_cast<size_t>(out_f),
                       static_cast<size_t>(in_f), 1);
    for (int i = 0; i < out_f; ++i) {
        y[i] += b[i];
    }
}

void linear_backward(const float* W, const float* x, const float* grad_y,
                     float* grad_W, float* grad_b, float* grad_x, int out_f,
                     int in_f) {
    for (int o = 0; o < out_f; ++o) {
        grad_b[o] += grad_y[o];
        for (int i = 0; i < in_f; ++i) {
            grad_W[o * in_f + i] += grad_y[o] * x[i];
            grad_x[i] += W[o * in_f + i] * grad_y[o];
        }
    }
}

Tensor Linear::forward(const Tensor& x) const {
    Tensor y = Tensor::zeros({out_});
    linear_forward(W_.data(), b_.data(), x.data(), y.data(), out_, in_);
    return y;
}

void Linear::backward(const Tensor& grad_out, const Tensor& x, Tensor& grad_W,
                      Tensor& grad_b, Tensor& grad_x) const {
    linear_backward(W_.data(), x.data(), grad_out.data(), grad_W.data(),
                    grad_b.data(), grad_x.data(), out_, in_);
}

bool Linear::save(const std::string& filepath) const {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    const uint32_t magic = 0x45525541; // Linear layer magic ('AURE')
    fwrite(&magic, sizeof(magic), 1, f);

    const uint32_t version = 1;
    fwrite(&version, sizeof(version), 1, f);

    fwrite(&in_, sizeof(in_), 1, f);
    fwrite(&out_, sizeof(out_), 1, f);

    fclose(f);

    if (!W_.save(filepath + ".W")) return false;
    if (!b_.save(filepath + ".b")) return false;

    return true;
}

Linear Linear::load(const std::string& filepath) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Could not open Linear file for loading: " + filepath);
    }

    uint32_t magic;
    fread(&magic, sizeof(magic), 1, f);
    if (magic != 0x45525541) {
        fclose(f);
        throw std::runtime_error("Invalid Aurelis Linear file");
    }

    uint32_t version;
    fread(&version, sizeof(version), 1, f);
    if (version != 1) {
        fclose(f);
        throw std::runtime_error("Unsupported Linear version");
    }

    int in, out;
    fread(&in, sizeof(in), 1, f);
    fread(&out, sizeof(out), 1, f);
    fclose(f);

    Linear linear(in, out);
    linear.W_ = Tensor::load(filepath + ".W");
    linear.b_ = Tensor::load(filepath + ".b");

    return linear;
}

}  // namespace aurelis::lens
