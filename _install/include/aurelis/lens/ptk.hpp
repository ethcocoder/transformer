#pragma once

#include "aurelis/scan.h"

namespace aurelis::lens {

inline int parallel_scan_forward(const float* a, const float* b,
                                 const float* h0, float* h_out, int n, int D) {
    return aurelis_scan_forward(a, b, h0, h_out, static_cast<size_t>(n),
                                static_cast<size_t>(D));
}

inline int parallel_scan_backward(const float* a, const float* h,
                                  const float* h0, const float* grad_h,
                                  float* grad_a, float* grad_b, float* grad_h0,
                                  int n, int D) {
    return aurelis_scan_backward(
        a, h, h0, grad_h, grad_a, grad_b, grad_h0, static_cast<size_t>(n),
        static_cast<size_t>(D));
}

}  // namespace aurelis::lens
