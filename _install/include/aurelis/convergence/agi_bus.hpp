#pragma once

#include "aurelis/convergence/common.hpp"

namespace aurelis::convergence {

class AgiBus {
public:
    AgiBus();

    Tensor read(int t, TaskType tau);
    void write(int t, TaskType tau, const Tensor& delta_down);
    void subscribe(TaskType tau, DownstreamCallback cb);

private:
    std::vector<DownstreamCallback> subscribers_world;
    std::vector<DownstreamCallback> subscribers_motor;
    std::vector<DownstreamCallback> subscribers_plan;
    std::vector<Tensor> history_world;
    std::vector<Tensor> history_motor;
    std::vector<Tensor> history_plan;
};

}  // namespace aurelis::convergence
