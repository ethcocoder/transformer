#include "aurelis/arc/config.hpp"
#include "aurelis/arc/ask_r.hpp"
#include "aurelis/lens/ietcf.hpp"
#include "aurelis/lens/fwse.hpp"
#include "aurelis/lens/csc.hpp"
#include "aurelis/lens/mgp.hpp"
#include "aurelis/lens/spi.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace aurelis::lens;
using namespace aurelis::arc;

static bool check_finite(const char* name, const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) {
            fprintf(stderr, "ERROR: NaN/Inf in %s at index %zu\n", name, i);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== ARC Phase 3 Test ===\n");

    LensConfig lens_cfg;
    lens_cfg.vocab_size = 16;
    lens_cfg.D = 64;
    lens_cfg.d_model = 32;
    lens_cfg.d_tau = 32;
    lens_cfg.d_ff = 256;
    lens_cfg.num_layers = 2;

    ArcConfig arc_cfg;
    arc_cfg.aureum_cfg.cfg = lens_cfg;
    const auto part = aurelis::StatePartition::from_total_dim(lens_cfg.D);

    // Create test sequence
    const int n = 12;
    std::vector<int> tokens(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        tokens[static_cast<size_t>(i)] = i % lens_cfg.vocab_size;
    }

    // Step 1: Run LENS to get c and gamma
    printf("1. Running LENS to get content c...\n");
    IETCF ietcf(lens_cfg);
    ietcf.init();
    std::vector<float> x_embed(static_cast<size_t>(n * lens_cfg.d_model));
    std::vector<float> gamma(static_cast<size_t>(n * lens_cfg.d_model));
    ietcf.forward(tokens.data(), n, x_embed.data(), gamma.data());

    FWSE fwse(lens_cfg);
    fwse.init();
    std::vector<float> a(static_cast<size_t>(n * lens_cfg.D));
    std::vector<float> b(static_cast<size_t>(n * lens_cfg.D));
    fwse.forward_sequence(x_embed.data(), gamma.data(), n, a.data(), b.data());

    CSC csc(lens_cfg);
    csc.init();
    std::vector<float> h_csc(static_cast<size_t>(n * lens_cfg.D));
    std::vector<float> h_raw(static_cast<size_t>(n * lens_cfg.D));
    csc.forward(a.data(), b.data(), n, h_csc.data(), h_raw.data());

    MGP mgp(lens_cfg);
    mgp.init();
    std::vector<float> h_mgp(static_cast<size_t>(n * lens_cfg.D));
    mgp.forward_sequence(h_csc.data(), h_mgp.data(), n);

    SPI spi(lens_cfg);
    spi.init();
    std::vector<float> c_lens(static_cast<size_t>(n * lens_cfg.d_model));
    std::vector<float> e_spi(static_cast<size_t>(n * part.De));
    std::vector<float> r_spi(static_cast<size_t>(n * part.Dr));
    std::vector<float> m_spi(static_cast<size_t>(n * part.Dm));
    float aux_spi = 0.0f;
    spi.forward_sequence(h_mgp.data(), c_lens.data(), e_spi.data(), r_spi.data(), m_spi.data(), n, aux_spi);

    // Step 2: Run full ARC 4-pass
    printf("2. Running ARC 4-pass forward...\n");
    ArcPassScheduler arc(arc_cfg);
    arc.init();

    std::vector<float> e(static_cast<size_t>(n * part.De));
    std::vector<float> r(static_cast<size_t>(n * part.Dr));
    std::vector<float> m(static_cast<size_t>(n * part.Dm));
    std::vector<float> logits(static_cast<size_t>(n * lens_cfg.vocab_size));
    float aux_loss = 0.0f;
    arc.forward(c_lens.data(), gamma.data(), x_embed.data(), n,
                e.data(), r.data(), m.data(), logits.data(), &aux_loss);

    // Check all outputs are finite
    bool all_ok = true;
    all_ok &= check_finite("e", e.data(), e.size());
    all_ok &= check_finite("r", r.data(), r.size());
    all_ok &= check_finite("m", m.data(), m.size());
    all_ok &= check_finite("logits", logits.data(), logits.size());

    if (all_ok) {
        printf("\n✓ All ARC outputs are finite!\n");
        printf("\n=== ARC Test PASSED ===\n");
        return 0;
    } else {
        printf("\n✗ ARC Test FAILED\n");
        return 1;
    }
}
