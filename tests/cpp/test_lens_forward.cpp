#include "aurelis/lens/lens_model.hpp"

#include <cmath>
#include <cstdio>
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

    LensModel model(cfg);
    model.init();

    const int n = 50;
    std::vector<int> tokens(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        tokens[static_cast<size_t>(i)] = i % cfg.vocab_size;
    }

    auto r1 = model.forward(tokens.data(), n);
    if (r1.logits.empty()) {
        fprintf(stderr, "forward produced no logits\n");
        return 1;
    }
    if (r1.loss_ce <= 0.0f) {
        fprintf(stderr, "expected positive CE loss\n");
        return 1;
    }

    float first = r1.loss_total;
    float last = first;
    for (int step = 0; step < 30; ++step) {
        last = model.train_step(tokens.data(), n).loss_total;
    }

    if (!std::isfinite(first) || !std::isfinite(last)) {
        fprintf(stderr, "loss is not finite: %f -> %f\n", first, last);
        return 1;
    }
    if (last >= first) {
        fprintf(stderr, "expected loss to decrease: %f -> %f\n", first, last);
        return 1;
    }

    printf("test_lens_forward: passed (loss %f -> %f)\n", first, last);
    return 0;
}
