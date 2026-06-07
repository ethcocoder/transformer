#include "aurelis/tesserae/epqe.hpp"

#include <cstring>
#include <algorithm>

namespace aurelis::tesserae {

EPQE::EPQE(const EPQEConfig& cfg) : cfg_(cfg) {}

int EPQE::select_bits(float competence) const {
    if (competence >= cfg_.mature_threshold) {
        return cfg_.mature_bits;
    }
    return cfg_.default_bits;
}

EPQE::QuantizedTensor EPQE::quantize(const Tensor& W, float competence) const {
    QuantizedTensor q;
    q.shape = W.shape();
    q.bits = select_bits(competence);
    int n = W.numel();

    float* w_data = const_cast<float*>(W.data());

    if (q.bits == 8) {
        q.data.resize(n);
        int8_t* q_data = reinterpret_cast<int8_t*>(q.data.data());
        aurelis_quantize_int8(w_data, q_data, &q.scale, &q.zero_point, n);
    } else if (q.bits == 4) {
        int num_bytes = (n + 1) / 2;
        q.data.resize(num_bytes);
        aurelis_quantize_int4(w_data, q.data.data(), &q.scale, &q.zero_point, n);
    } else {
        // Fallback to FP32 (store as raw bytes)
        q.bits = 32;
        q.data.resize(n * sizeof(float));
        std::memcpy(q.data.data(), w_data, n * sizeof(float));
        q.scale = 1.0f;
        q.zero_point = 0.0f;
    }

    return q;
}

Tensor EPQE::dequantize(const QuantizedTensor& q) const {
    Tensor W = Tensor::zeros(q.shape);
    float* w_data = W.data();
    int n = W.numel();

    if (q.bits == 8) {
        const int8_t* q_data = reinterpret_cast<const int8_t*>(q.data.data());
        aurelis_dequantize_int8(q_data, w_data, q.scale, q.zero_point, n);
    } else if (q.bits == 4) {
        aurelis_dequantize_int4(q.data.data(), w_data, q.scale, q.zero_point, n);
    } else if (q.bits == 32) {
        std::memcpy(w_data, q.data.data(), n * sizeof(float));
    }

    return W;
}

} // namespace aurelis::tesserae
