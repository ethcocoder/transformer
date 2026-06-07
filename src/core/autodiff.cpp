#include "aurelis/autodiff.hpp"

#include "aurelis/matmul.h"

#include <cstring>
#include <stdexcept>

namespace aurelis {

int Tape::emplace(const Tensor& value, OpType op,
                  const std::vector<int>& parents) {
    Node n;
    n.op = op;
    n.parents = parents;
    n.value = value;
    n.grad = Tensor::zeros(value.shape());
    nodes_.push_back(std::move(n));
    return static_cast<int>(nodes_.size()) - 1;
}

void Tape::clear() { nodes_.clear(); }

void Tape::backward(int root_id, const Tensor* seed_grad) {
    if (root_id < 0 || root_id >= size()) {
        throw std::out_of_range("backward: invalid root id");
    }

    for (auto& n : nodes_) {
        n.grad.fill(0.0f);
    }

    if (seed_grad != nullptr) {
        if (seed_grad->numel() != nodes_[root_id].value.numel()) {
            throw std::invalid_argument("backward: seed_grad shape mismatch");
        }
        for (int64_t j = 0; j < seed_grad->numel(); ++j) {
            nodes_[root_id].grad.at(j) = seed_grad->at(j);
        }
    } else {
        nodes_[root_id].grad.fill(1.0f);
    }

    for (int i = root_id; i >= 0; --i) {
        Node& n = nodes_[i];
        if (n.grad.numel() == 0) {
            continue;
        }

        switch (n.op) {
            case OpType::Add: {
                if (n.parents.size() != 2) {
                    break;
                }
                for (int p : n.parents) {
                    for (int64_t j = 0; j < n.grad.numel(); ++j) {
                        nodes_[p].grad.at(j) += n.grad.at(j);
                    }
                }
                break;
            }
            case OpType::Mul: {
                if (n.parents.size() != 2) {
                    break;
                }
                const int a = n.parents[0];
                const int b = n.parents[1];
                for (int64_t j = 0; j < n.grad.numel(); ++j) {
                    nodes_[a].grad.at(j) += n.grad.at(j) * nodes_[b].value.at(j);
                    nodes_[b].grad.at(j) += n.grad.at(j) * nodes_[a].value.at(j);
                }
                break;
            }
            case OpType::Matmul: {
                if (n.parents.size() != 2) {
                    break;
                }
                const int a_id = n.parents[0];
                const int b_id = n.parents[1];
                const Tensor& A = nodes_[a_id].value;
                const Tensor& B = nodes_[b_id].value;
                const size_t M = static_cast<size_t>(A.shape()[0]);
                const size_t K = static_cast<size_t>(A.shape()[1]);
                const size_t N = static_cast<size_t>(B.shape()[1]);

                Tensor grad_A = Tensor::zeros(A.shape());
                Tensor grad_B = Tensor::zeros(B.shape());

                /* grad_A = grad_C @ B^T */
                aurelis_matmul_backward_f32(A.data(), B.data(), n.grad.data(),
                                            grad_A.data(), grad_B.data(), M, K,
                                            N);

                for (int64_t j = 0; j < grad_A.numel(); ++j) {
                    nodes_[a_id].grad.at(j) += grad_A.at(j);
                }
                for (int64_t j = 0; j < grad_B.numel(); ++j) {
                    nodes_[b_id].grad.at(j) += grad_B.at(j);
                }
                break;
            }
            default:
                break;
        }
    }
}

int tape_add(Tape& tape, int a_id, int b_id) {
    const Tensor& a = tape.node(a_id).value;
    const Tensor& b = tape.node(b_id).value;
    Tensor out = Tensor::zeros(a.shape());
    add(a, b, out);
    return tape.emplace(out, OpType::Add, {a_id, b_id});
}

int tape_matmul(Tape& tape, int a_id, int b_id) {
    const Tensor& A = tape.node(a_id).value;
    const Tensor& B = tape.node(b_id).value;
    Tensor C = Tensor::zeros({A.shape()[0], B.shape()[1]});
    matmul(A, B, C);
    return tape.emplace(C, OpType::Matmul, {a_id, b_id});
}

}  // namespace aurelis
