#pragma once

#include <aurelis/tensor.hpp>
#include <functional>
#include <vector>

namespace aurelis::convergence {

enum class TaskType {
    World,
    Motor,
    Plan
};

using DownstreamCallback = std::function<void(TaskType, const Tensor&)>;

struct EpistemicFrame {
    Tensor c;
    Tensor e;
    Tensor d;
    Tensor alpha;
    float kappa;
    Tensor sigma;
    float H_reason;
};

}  // namespace aurelis::convergence
