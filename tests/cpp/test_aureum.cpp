#include "aurelis/aureum/config.hpp"
#include "aurelis/aureum/pii.hpp"
#include "aurelis/lens/ietcf.hpp"
#include "aurelis/lens/fwse.hpp"
#include "aurelis/lens/csc.hpp"
#include "aurelis/lens/mgp.hpp"
#include "aurelis/lens/spi.hpp"
#include "aurelis/lens/osh.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace aurelis::lens;
using namespace aurelis::aureum;

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
    printf("=== AUREUM Phase 2 Test ===\n");

    LensConfig lens_cfg;
    lens_cfg.vocab_size = 16;
    lens_cfg.D = 64;
    lens_cfg.d_model = 32;
    lens_cfg.d_tau = 32;
    lens_cfg.d_ff = 256;
    lens_cfg.num_layers = 2;

    AureumConfig aureum_cfg;
    aureum_cfg.cfg = lens_cfg;
    const auto part = aurelis::StatePartition::from_total_dim(lens_cfg.D);

    // Create test sequence
    const int n = 12;
    std::vector<int> tokens(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        tokens[static_cast<size_t>(i)] = i % lens_cfg.vocab_size;
    }

    // Step 1: Run LENS to get c
    printf("1. Running LENS to get content c...\n");
    IETCF ietcf(lens_cfg);
    ietcf.init();
    std::vector<float> x(static_cast<size_t>(n * lens_cfg.d_model));
    std::vector<float> gamma(static_cast<size_t>(n * lens_cfg.d_model));
    ietcf.forward(tokens.data(), n, x.data(), gamma.data());

    FWSE fwse(lens_cfg);
    fwse.init();
    std::vector<float> a(static_cast<size_t>(n * lens_cfg.D));
    std::vector<float> b(static_cast<size_t>(n * lens_cfg.D));
    fwse.forward_sequence(x.data(), gamma.data(), n, a.data(), b.data());

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
    float aux = 0.0f;
    spi.forward_sequence(h_mgp.data(), c_lens.data(), e_spi.data(), r_spi.data(), m_spi.data(), n, aux);

    // Step 2: Run AUREUM 3-pass
    printf("2. Running AUREUM 3-pass forward...\n");
    AureumPassScheduler scheduler(aureum_cfg);
    scheduler.init();

    std::vector<float> e_aureum(static_cast<size_t>(n * part.De));
    std::vector<float> kappa(static_cast<size_t>(n));
    std::vector<float> m_aureum(static_cast<size_t>(n * part.Dm));
    std::vector<float> c_mod(static_cast<size_t>(n * lens_cfg.d_model));

    scheduler.forward(c_lens.data(), n, e_aureum.data(), kappa.data(), m_aureum.data(), c_mod.data());

    // Check all outputs are finite
    bool all_ok = true;
    all_ok &= check_finite("e_aureum", e_aureum.data(), e_aureum.size());
    all_ok &= check_finite("kappa", kappa.data(), kappa.size());
    all_ok &= check_finite("m_aureum", m_aureum.data(), m_aureum.size());
    all_ok &= check_finite("c_mod", c_mod.data(), c_mod.size());

    if (all_ok) {
        printf("\n✓ All AUREUM outputs are finite!\n");
        printf("  Kappa range: [%.4f, %.4f]\n",
               *std::min_element(kappa.begin(), kappa.end()),
               *std::max_element(kappa.begin(), kappa.end()));
        printf("\n=== AUREUM Test PASSED ===\n");
        return 0;
    } else {
        printf("\n✗ AUREUM Test FAILED\n");
        return 1;
    }
}
