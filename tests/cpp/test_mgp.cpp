#include "aurelis/lens/config.hpp"
#include "aurelis/lens/mgp.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace aurelis::lens;

int main() {
    LensConfig cfg;
    cfg.D = 32;
    MGP mgp(cfg);
    mgp.init();

    std::vector<float> h_in(static_cast<size_t>(cfg.D), 1.0f);
    std::vector<float> h_out(static_cast<size_t>(cfg.D));
    mgp.forward(h_in.data(), h_out.data());

    std::vector<float> centered(static_cast<size_t>(cfg.D));
    for (int i = 0; i < cfg.D; ++i) {
        centered[static_cast<size_t>(i)] = h_out[static_cast<size_t>(i)];
    }
    float norm = 0.0f;
    for (int i = 0; i < cfg.D; ++i) {
        norm += centered[static_cast<size_t>(i)] * centered[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm / static_cast<float>(cfg.D));
    if (std::fabs(norm - 1.0f) > 0.2f) {
        fprintf(stderr, "MGP norm expected ~1, got %f\n", norm);
        return 1;
    }

    printf("test_mgp: passed\n");
    return 0;
}
