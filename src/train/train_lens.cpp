#include "aurelis/lens/lens_model.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace aurelis::lens;

int main() {
    LensConfig cfg;
    cfg.vocab_size = 16;
    cfg.D = 64;
    cfg.d_model = 32;
    cfg.d_tau = 32;
    cfg.d_ff = 256;
    cfg.num_layers = 2;
    cfg.lr = 0.01f;

    LensModel model(cfg);
    model.init();

    const std::string corpus = "aurelis lens o n sequence model phase one training";
    std::vector<int> tokens;
    for (char ch : corpus) {
        tokens.push_back(static_cast<int>(ch & 15));
    }
    const int n = static_cast<int>(tokens.size());

    printf("Aurelis LENS Phase I training demo\n");
    printf("tokens=%d layers=%d D=%d\n", n, cfg.num_layers, cfg.D);

    for (int epoch = 0; epoch < 50; ++epoch) {
        auto r = model.train_step(tokens.data(), n);
        if (epoch % 10 == 0) {
            printf("epoch %2d  loss=%.4f  ce=%.4f  aux=%.4f  stab=%.4f\n", epoch,
                   r.loss_total, r.loss_ce, r.loss_aux, r.loss_stab);
        }
    }

    auto final_r = model.forward(tokens.data(), n);
    printf("done  final_ce=%.4f\n", final_r.loss_ce);
    return 0;
}
