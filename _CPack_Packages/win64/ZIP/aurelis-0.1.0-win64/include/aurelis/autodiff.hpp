#pragma once

#include "aurelis/tensor.hpp"

#include <functional>
#include <vector>

namespace aurelis {

enum class OpType { Leaf, Add, Mul, Matmul, Scan };

struct Node {
    OpType op;
    std::vector<int> parents;
    Tensor value;
    Tensor grad;
};

class Tape {
 public:
    int emplace(const Tensor& value, OpType op,
                const std::vector<int>& parents);
    void backward(int root_id, const Tensor* seed_grad = nullptr);
    void clear();

    const Node& node(int id) const { return nodes_.at(id); }
    Node& node(int id) { return nodes_.at(id); }
    int size() const { return static_cast<int>(nodes_.size()); }

 private:
    std::vector<Node> nodes_;
};

int tape_add(Tape& tape, int a_id, int b_id);
int tape_matmul(Tape& tape, int a_id, int b_id);

}  // namespace aurelis
