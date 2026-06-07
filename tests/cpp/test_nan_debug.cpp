#include "aurelis/lens/csc.hpp"
#include "aurelis/lens/fwse.hpp"
#include "aurelis/lens/ietcf.hpp"
#include "aurelis/lens/lens_layer.hpp"
#include "aurelis/lens/mgp.hpp"
#include "aurelis/lens/osh.hpp"
#include "aurelis/lens/spi.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace aurelis::lens;

static bool check(const char* name, const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) {
            fprintf(stderr, "NAN in %s at %zu val=%f\n", name, i, p[i]);
            return false;
        }
    }
    return true;
}

int main() {
    LensConfig cfg;
    cfg.vocab_size = 16;
    cfg.D = 64;
    cfg.d_model = 32;
    cfg.d_tau = 32;
    cfg.d_ff = 256;
    cfg.num_layers = 2;

    const int n = 12;
    std::vector<int> tokens(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        tokens[static_cast<size_t>(i)] = i % cfg.vocab_size;
    }

    IETCF ietcf(cfg);
    ietcf.init();
    std::vector<float> x(static_cast<size_t>(n * cfg.d_model));
    std::vector<float> gamma(static_cast<size_t>(n * cfg.d_model));
    ietcf.forward(tokens.data(), n, x.data(), gamma.data());
    if (!check("x_embed", x.data(), x.size())) {
        return 1;
    }
    if (!check("gamma", gamma.data(), gamma.size())) {
        return 1;
    }

    FWSE fwse(cfg);
    fwse.init();
    std::vector<float> a(static_cast<size_t>(n * cfg.D));
    std::vector<float> b(static_cast<size_t>(n * cfg.D));
    fwse.forward_sequence(x.data(), gamma.data(), n, a.data(), b.data());
    if (!check("a", a.data(), a.size())) {
        return 1;
    }
    if (!check("b", b.data(), b.size())) {
        return 1;
    }

    CSC csc(cfg);
    csc.init();
    std::vector<float> h_csc(static_cast<size_t>(n * cfg.D));
    std::vector<float> h_raw(static_cast<size_t>(n * cfg.D));
    csc.forward(a.data(), b.data(), n, h_csc.data(), h_raw.data());
    if (!check("h_csc", h_csc.data(), h_csc.size())) {
        return 1;
    }

    MGP mgp(cfg);
    mgp.init();
    std::vector<float> h_mgp(static_cast<size_t>(n * cfg.D));
    mgp.forward_sequence(h_csc.data(), h_mgp.data(), n);
    if (!check("h_mgp", h_mgp.data(), h_mgp.size())) {
        return 1;
    }

    SPI spi(cfg);
    spi.init();
    const auto& part = spi.partition();
    std::vector<float> c(static_cast<size_t>(n * cfg.d_model));
    std::vector<float> e(static_cast<size_t>(n * part.De));
    std::vector<float> r(static_cast<size_t>(n * part.Dr));
    std::vector<float> m(static_cast<size_t>(n * part.Dm));
    float aux = 0.0f;
    spi.forward_sequence(h_mgp.data(), c.data(), e.data(), r.data(), m.data(), n,
                         aux);
    if (!check("c", c.data(), c.size())) {
        return 1;
    }
    printf("aux=%f\n", aux);

    OSH osh(cfg);
    osh.init();
    std::vector<float> logits(static_cast<size_t>(n * cfg.vocab_size));
    osh.forward_sequence(c.data(), x.data(), n, logits.data());
    if (!check("logits", logits.data(), logits.size())) {
        return 1;
    }

    printf("all components finite\n");
    return 0;
}
