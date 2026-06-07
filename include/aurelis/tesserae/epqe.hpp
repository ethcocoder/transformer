#pragma once

#include <aurelis/tensor.hpp>
#include <aurelis/c/quant_int4_int8.h>
#include <vector>

namespace aurelis::tesserae {

struct EPQEConfig {
    float mature_threshold;    // Threshold for mature domains (CGFC: Conservative Gradient Freezing Condition)
    int default_bits;          // Default quantization bits (8)
    int mature_bits;           // Quantization bits for mature domains (4)
};

class EPQE {
public:
    explicit EPQE(const EPQEConfig& cfg);

    // Quantize a tensor, returning quantized data and metadata
    struct QuantizedTensor {
        std::vector<uint8_t> data;
        float scale;
        float zero_point;
        int bits;
        std::vector<int64_t> shape;
    };

    QuantizedTensor quantize(const Tensor& W, float competence = 1.0f) const;

    // Dequantize back to float
    Tensor dequantize(const QuantizedTensor& q) const;

    // DMAS: Domain-aware Multi-bit Allocation Strategy
    int select_bits(float competence) const;

private:
    EPQEConfig cfg_;
};

} // namespace aurelis::tesserae
