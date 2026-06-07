#include "aurelis/lens/config.hpp"
#include "aurelis/lens/fwse.hpp"

#include <cstdio>
#include <vector>

using namespace aurelis::lens;

int main() {
    LensConfig cfg;
    cfg.D = 64;
    cfg.d_model = 32;
    cfg.d_ff = 256;

    FWSE fwse(cfg);
    fwse.init();

    std::vector<float> x(static_cast<size_t>(cfg.d_model), 0.1f);
    std::vector<float> g(static_cast<size_t>(cfg.d_model), 0.05f);
    std::vector<float> a(static_cast<size_t>(cfg.D));
    std::vector<float> b(static_cast<size_t>(cfg.D));

    fwse.forward_step(x.data(), g.data(), a.data(), b.data());

    for (int i = 0; i < cfg.D; ++i) {
        if (a[static_cast<size_t>(i)] >= 1.0f) {
            fprintf(stderr, "forget gate must be < 1, got %f\n",
                    a[static_cast<size_t>(i)]);
            return 1;
        }
        if (a[static_cast<size_t>(i)] < 0.0f) {
            fprintf(stderr, "forget gate must be >= 0\n");
            return 1;
        }
    }

    printf("test_fwse_spectral: passed\n");
    return 0;
}
