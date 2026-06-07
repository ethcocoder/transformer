#include "aurelis/lens/lens_layer.hpp"

#include <cstring>

namespace aurelis::lens {

LensLayer::LensLayer(const LensConfig& cfg)
    : cfg_(cfg), fwse_(cfg), csc_(cfg), mgp_(cfg), spi_(cfg) {}

void LensLayer::init() {
    fwse_.init();
    csc_.init();
    mgp_.init();
    spi_.init();
}

void LensLayer::forward(const float* x_stream, const float* gamma, int n,
                        float* x_out, LayerCache& cache) const {
    const int D = cfg_.D;
    const int Dc = cfg_.Dc();
    const int De = cfg_.De();
    const int Dr = cfg_.Dr();
    const int Dm = cfg_.Dm();

    cache.a.resize(static_cast<size_t>(n * D));
    cache.b.resize(static_cast<size_t>(n * D));
    cache.h_raw.resize(static_cast<size_t>(n * D));
    cache.h_csc.resize(static_cast<size_t>(n * D));
    cache.h_mgp.resize(static_cast<size_t>(n * D));
    cache.c.resize(static_cast<size_t>(n * Dc));
    cache.e.resize(static_cast<size_t>(n * De));
    cache.r.resize(static_cast<size_t>(n * Dr));
    cache.m.resize(static_cast<size_t>(n * Dm));

    fwse_.forward_sequence(x_stream, gamma, n, cache.a.data(), cache.b.data());
    csc_.forward(cache.a.data(), cache.b.data(), n, cache.h_csc.data(),
                 cache.h_raw.data());
    mgp_.forward_sequence(cache.h_csc.data(), cache.h_mgp.data(), n);
    spi_.forward_sequence(cache.h_mgp.data(), cache.c.data(), cache.e.data(),
                          cache.r.data(), cache.m.data(), n, cache.aux_loss);

    for (int t = 0; t < n; ++t) {
        for (int d = 0; d < Dc; ++d) {
            x_out[t * Dc + d] =
                cache.c[t * Dc + d] + x_stream[t * Dc + d];
        }
    }
}

}  // namespace aurelis::lens
