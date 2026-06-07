#include "aurelis/lens/ietcf.hpp"
#include "aurelis/lens/linear.hpp"
#include "aurelis/orthogonal.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace aurelis;
using namespace aurelis::lens;

int main() {
    LensConfig cfg;
    cfg.d_model = 32;
    cfg.d_tau = 32;
    cfg.vocab_size = 16;

    IETCF ietcf(cfg);
    ietcf.init();

    float* R = ietcf.rotation().data();
    for (int i = 0; i < 32 * 32; ++i) {
        if (!std::isfinite(R[i])) {
            printf("R has nan at %d\n", i);
            return 1;
        }
    }

    std::vector<float> tau(32, 0.0f);
    tau[0] = 1.0f;
    std::vector<float> tau_new(32);
    aurelis_orthogonal_apply(R, tau.data(), tau_new.data(), 32);
    if (!std::isfinite(tau_new[0])) {
        printf("tau_new nan: %f\n", tau_new[0]);
        return 1;
    }

    Linear W(cfg.d_tau, cfg.d_model);
    W.init_xavier();
    for (int64_t i = 0; i < W.weight().numel(); ++i) {
        if (!std::isfinite(W.weight().at(i))) {
            printf("W nan at %lld\n", (long long)i);
            return 1;
        }
    }

    Tensor tau_t = Tensor::from_data({32}, tau_new.data(), false);
    Tensor g = W.forward(tau_t);
    printf("gamma[0]=%f\n", g.at(0));
    return std::isfinite(g.at(0)) ? 0 : 1;
}
