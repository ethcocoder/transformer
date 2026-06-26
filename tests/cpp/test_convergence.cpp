#include "aurelis/convergence/efe.hpp"
#include "aurelis/convergence/agi_bus.hpp"
#include <iostream>
#include <vector>

namespace {
bool require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}
}  // namespace

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

    for (int i = 0; i < static_cast<int>(shape_c[0]); ++i) c.at(i) = static_cast<float>(i + 1);
    for (int i = 0; i < static_cast<int>(shape_e[0]); ++i) e.at(i) = 0.25f * (i + 1);
    for (int i = 0; i < static_cast<int>(shape_d[0]); ++i) d.at(i) = 0.5f * (i + 1);
    for (int i = 0; i < static_cast<int>(shape_alpha[0]); ++i) alpha.at(i) = 1.0f + i;
    for (int i = 0; i < static_cast<int>(shape_sigma[0]); ++i) sigma.at(i) = 0.1f * (i + 1);

    EpistemicFrame frame = efe.assemble(c, e, d, alpha, 0.5f, sigma, 1.0f);

    // Test encoding
    Tensor enc = efe.encode_frame(frame);
    if (!require(enc.shape() == std::vector<int64_t>{32}, "encode_frame must return D_bus-sized output")) {
        return 1;
    }

    float sum = 0.0f;
    for (int i = 0; i < enc.shape()[0]; ++i) {
        sum += enc.at(i);
    }
    if (!require(sum != 0.0f, "encode_frame should produce a non-trivial frame summary")) {
        return 1;
    }

    std::cout << "Frame encoded successfully!\n";

    // Test adapter outputs
    auto adapters = efe.get_adapter_outputs(enc);
    std::cout << "Adapter outputs created: " << adapters.size() << "\n";

    std::cout << "Phase 5 tests passed!\n";
    return 0;
}
