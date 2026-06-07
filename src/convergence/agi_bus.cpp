#include "aurelis/convergence/agi_bus.hpp"

namespace aurelis::convergence {

AgiBus::AgiBus() {}

Tensor AgiBus::read(int t, TaskType tau) {
    switch (tau) {
        case TaskType::World:
            if (t < (int)history_world.size()) {
                return history_world[t];
            }
            break;
        case TaskType::Motor:
            if (t < (int)history_motor.size()) {
                return history_motor[t];
            }
            break;
        case TaskType::Plan:
            if (t < (int)history_plan.size()) {
                return history_plan[t];
            }
            break;
    }
    return Tensor::zeros({1});
}

void AgiBus::write(int t, TaskType tau, const Tensor& delta_down) {
    (void)t;
    (void)tau;
    (void)delta_down;
    // TODO: implement writing to history
}

void AgiBus::subscribe(TaskType tau, DownstreamCallback cb) {
    switch (tau) {
        case TaskType::World:
            subscribers_world.push_back(cb);
            break;
        case TaskType::Motor:
            subscribers_motor.push_back(cb);
            break;
        case TaskType::Plan:
            subscribers_plan.push_back(cb);
            break;
    }
}

}  // namespace aurelis::convergence
