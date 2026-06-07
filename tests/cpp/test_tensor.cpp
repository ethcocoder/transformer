#include "aurelis/state_partition.hpp"
#include "aurelis/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace aurelis;

static int test_tensor_basic() {
    Tensor t = Tensor::zeros({3, 4});
    if (t.numel() != 12 || t.ndim() != 2) {
        return 1;
    }
    t.at(0) = 1.0f;
  Tensor v = t.view({4, 3});
    if (v.numel() != 12) {
        return 1;
    }
    return 0;
}

static int test_state_partition_roundtrip() {
    const int D = 100;
    StatePartition p = StatePartition::from_total_dim(D);
    if (p.Dc + p.De + p.Dr + p.Dm != D) {
        fprintf(stderr, "partition dims don't sum to D\n");
        return 1;
    }

    std::vector<float> h(D), c(p.Dc), e(p.De), r(p.Dr), m(p.Dm), h2(D);
    for (int i = 0; i < D; ++i) {
        h[i] = static_cast<float>(i);
    }
    p.split(h.data(), c.data(), e.data(), r.data(), m.data());
    p.merge(h2.data(), c.data(), e.data(), r.data(), m.data());

    for (int i = 0; i < D; ++i) {
        if (h[i] != h2[i]) {
            fprintf(stderr, "roundtrip failed at %d\n", i);
            return 1;
        }
    }
    return 0;
}

static int test_matmul() {
    Tensor A = Tensor::zeros({2, 3});
    Tensor B = Tensor::zeros({3, 4});
    Tensor C = Tensor::zeros({2, 4});
    for (int i = 0; i < A.numel(); ++i) {
        A.at(i) = 0.1f * static_cast<float>(i);
    }
    for (int i = 0; i < B.numel(); ++i) {
        B.at(i) = 0.05f * static_cast<float>(i + 1);
    }
    matmul(A, B, C);
    if (std::fabs(C.at(0)) < 1e-6f) {
        return 1;
    }
    return 0;
}

int main() {
    if (test_tensor_basic() != 0) {
        fprintf(stderr, "test_tensor_basic failed\n");
        return 1;
    }
    if (test_state_partition_roundtrip() != 0) {
        fprintf(stderr, "test_state_partition_roundtrip failed\n");
        return 1;
    }
    if (test_matmul() != 0) {
        fprintf(stderr, "test_matmul failed\n");
        return 1;
    }
    printf("test_tensor_cpp: all passed\n");
    return 0;
}
