#include "aurelis/autodiff.hpp"
#include "aurelis/spectral.hpp"
#include "aurelis/tensor.hpp"

#include <cmath>
#include <cstdio>

using namespace aurelis;

static int test_autodiff_matmul() {
    Tensor A = Tensor::zeros({2, 3}, true);
    Tensor B = Tensor::zeros({3, 2}, true);
    for (int i = 0; i < A.numel(); ++i) {
        A.at(i) = 0.2f * static_cast<float>(i + 1);
    }
    for (int i = 0; i < B.numel(); ++i) {
        B.at(i) = 0.1f + 0.05f * static_cast<float>(i);
    }

    Tape tape;
    const int a_id = tape.emplace(A, OpType::Leaf, {});
    const int b_id = tape.emplace(B, OpType::Leaf, {});
    const int c_id = tape_matmul(tape, a_id, b_id);

    Tensor seed = Tensor::zeros({2, 2});
    seed.at(0) = 1.0f;
    tape.backward(c_id, &seed);

    const float eps = 1e-3f;

    auto loss_of = [&](const Tensor& AA) {
        Tensor CC = Tensor::zeros({2, 2});
        matmul(AA, B, CC);
        return CC.at(0);
    };

    for (int i = 0; i < A.numel(); ++i) {
        Tensor Ap = A.clone();
        Tensor Am = A.clone();
        Ap.at(i) += eps;
        Am.at(i) -= eps;
        const float num = (loss_of(Ap) - loss_of(Am)) / (2.0f * eps);
        const float ana = tape.node(a_id).grad.at(i);
        if (std::fabs(num - ana) > 0.05f) {
            fprintf(stderr, "autodiff A grad mismatch at %d: num=%f ana=%f\n", i,
                    num, ana);
            return 1;
        }
    }
    return 0;
}

static int test_spectral_clamp() {
    const float raw[4] = {0.0f, 1.0f, -1.0f, 5.0f};
    float out[4];
    const float eps[1] = {0.1f};
    const int scale[4] = {0, 0, 0, 0};
    clamp_forget_gate(raw, out, 4, eps, scale, 1);
    for (int i = 0; i < 4; ++i) {
        if (out[i] < 0.0f || out[i] > 0.9f) {
            fprintf(stderr, "spectral clamp out of range: %f\n", out[i]);
            return 1;
        }
    }
    return 0;
}

int main() {
    if (test_spectral_clamp() != 0) {
        return 1;
    }
    if (test_autodiff_matmul() != 0) {
        return 1;
    }
    printf("test_phase0_cpp: all passed\n");
    return 0;
}
