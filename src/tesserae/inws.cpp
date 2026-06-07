#include "aurelis/tesserae/inws.hpp"
#include "aurelis/lens/activations.hpp"
#include <cstring>
#include <cmath>

namespace aurelis::tesserae {

INWS::INWS(const INWSConfig& cfg)
    : cfg_(cfg),
      fc1_(cfg.latent_dim + cfg.pe_dim + cfg.layer_type_embed, cfg.hidden_dim),
      fc2_(cfg.hidden_dim, cfg.hidden_dim),
      fc3_(cfg.hidden_dim, 1) {}

void INWS::init() {
    fc1_.init_xavier();
    fc2_.init_xavier();
    fc3_.init_xavier();
}

Tensor INWS::synthesize(const Tensor& z, const Tensor& pe_i, const Tensor& t_emb) const {
    // Concatenate inputs: [z; pe_i; t_emb]
    std::vector<int64_t> input_shape = {
        static_cast<int64_t>(cfg_.latent_dim + cfg_.pe_dim + cfg_.layer_type_embed)
    };
    Tensor input = Tensor::zeros(input_shape);
    float* data = input.data();

    // Copy z
    std::memcpy(data, z.data(), cfg_.latent_dim * sizeof(float));
    // Copy pe_i
    std::memcpy(data + cfg_.latent_dim, pe_i.data(), cfg_.pe_dim * sizeof(float));
    // Copy t_emb
    std::memcpy(data + cfg_.latent_dim + cfg_.pe_dim, t_emb.data(), cfg_.layer_type_embed * sizeof(float));

    // Forward pass through hypernetwork
    Tensor h1 = fc1_.forward(input);
    lens::silu_forward(h1.data(), h1.data(), h1.numel());

    Tensor h2 = fc2_.forward(h1);
    lens::silu_forward(h2.data(), h2.data(), h2.numel());

    Tensor w_i = fc3_.forward(h2);
    return w_i;
}

Tensor INWS::synthesize_full(const Tensor& z, const Tensor& t_emb, int out_dim, int in_dim) const {
    int num_params = out_dim * in_dim;
    std::vector<int64_t> weight_shape = {static_cast<int64_t>(out_dim), static_cast<int64_t>(in_dim)};
    Tensor weights = Tensor::zeros(weight_shape);
    float* weight_data = weights.data();

    // Generate positional encodings for each parameter index
    for (int i = 0; i < num_params; ++i) {
        // Sinusoidal positional encoding
        std::vector<int64_t> pe_shape = {static_cast<int64_t>(cfg_.pe_dim)};
        Tensor pe_i = Tensor::zeros(pe_shape);
        float* pe_data = pe_i.data();
        for (int j = 0; j < cfg_.pe_dim; ++j) {
            if (j % 2 == 0) {
                pe_data[j] = std::sin(static_cast<float>(i) / std::pow(10000.0f, static_cast<float>(j) / static_cast<float>(cfg_.pe_dim)));
            } else {
                pe_data[j] = std::cos(static_cast<float>(i) / std::pow(10000.0f, static_cast<float>(j-1) / static_cast<float>(cfg_.pe_dim)));
            }
        }

        // Synthesize weight i
        Tensor w_i = synthesize(z, pe_i, t_emb);
        weight_data[i] = w_i.data()[0];
    }

    return weights;
}

} // namespace aurelis::tesserae
