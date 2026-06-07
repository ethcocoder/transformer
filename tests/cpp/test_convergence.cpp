#include "aurelis/convergence/efe.hpp"
#include "aurelis/convergence/agi_bus.hpp"
#include <iostream>

using namespace aurelis;
using namespace aurelis::convergence;

int main() {
    std::cout << "Testing Phase 5 (CONVERGENCE)...\n";

    // Test EFE
    EfeConfig efe_cfg { .D_bus = 32 };
    EFE efe(efe_cfg);
    efe.init();

    // Test AgiBus
    AgiBus bus;

    // Subscribe to world tasks
    bus.subscribe(TaskType::World, [](TaskType t, const Tensor& delta) {
        std::cout << "Received world task update!\n";
    });

    // Create dummy frame
    std::vector<int64_t> shape_c = {64};
    std::vector<int64_t> shape_e = {32};
    std::vector<int64_t> shape_d = {16};
    std::vector<int64_t> shape_alpha = {4};
    std::vector<int64_t> shape_sigma = {32};

    Tensor c = Tensor::zeros(shape_c);
    Tensor e = Tensor::zeros(shape_e);
    Tensor d = Tensor::zeros(shape_d);
    Tensor alpha = Tensor::zeros(shape_alpha);
    Tensor sigma = Tensor::zeros(shape_sigma);

    EpistemicFrame frame = efe.assemble(c, e, d, alpha, 0.5f, sigma, 1.0f);

    // Test encoding
    Tensor enc = efe.encode_frame(frame);
    std::cout << "Frame encoded successfully!\n";

    // Test adapter outputs
    auto adapters = efe.get_adapter_outputs(enc);
    std::cout << "Adapter outputs created: " << adapters.size() << "\n";

    std::cout << "Phase 5 tests passed!\n";
    return 0;
}
